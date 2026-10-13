"""검색 원장을 보존하는 공통 자료 수집기. 평가 점수는 생성하지 않는다."""
import hashlib
from copy import deepcopy
import json
import os
import re
from urllib.request import Request, urlopen
from .contracts import Collection, Evidence, Finding, Contract
from .evaluation_support import create_model_from_env
from pydantic import Field, ValidationError, create_model
from typing import Literal


class GeneratedExcerpt(Contract):
    segment_id: str = Field(min_length=1)


class GeneratedFinding(Contract):
    topic: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    excerpts: list[GeneratedExcerpt] = Field(min_length=1)


class FindingsOutput(Contract):
    findings: list[GeneratedFinding]
    missing_items: list[str] = Field(default_factory=list)



def safe_error_details(exc):
    if isinstance(exc, ValidationError):
        return [{'field': '.'.join(str(v) for v in e['loc']), 'type': e['type'], 'message': e['msg'][:200]} for e in exc.errors(include_input=False, include_context=False, include_url=False)]
    return []


def source_segments(registry):
    """모델은 실제 원문 구간을 선택하고, ID와 발췌는 코드가 연결한다."""
    lookup, documents = {}, []
    for eid, evidence in registry.items():
        candidates = []
        for match in re.finditer(r'[^\n。!?]+[。!?]?', evidence.excerpt):
            text = match.group().strip()
            if len(text) < 20:
                continue
            candidates.append(text[:400])
            if len(candidates) == 3:
                break
        if not candidates and evidence.excerpt.strip():
            candidates = [evidence.excerpt.strip()[:400]]
        segments = []
        for text in candidates:
            ident = f's{len(lookup) + 1}'
            lookup[ident] = (eid, text)
            segments.append({'segment_id': ident, 'text': text})
        document = evidence.model_dump()
        document.pop('excerpt')
        document['segments'] = segments
        documents.append(document)
    return lookup, documents


def extraction_schema(segment_ids):
    citation = create_model('SourceSelection', __base__=GeneratedExcerpt,
                            segment_id=(Literal[tuple(segment_ids)], ...))
    finding = create_model('SelectedFinding', __base__=GeneratedFinding, excerpts=(list[citation], Field(min_length=1)))
    return create_model('FindingsOutput', __base__=FindingsOutput, findings=(list[finding], ...))


def validated_findings(items, registry, segments):
    accepted, issues = [], []
    for index, item in enumerate(items):
        excerpts = {}
        for selection in item.excerpts:
            if selection.segment_id not in segments:
                issues.append(f'조사 항목 {index + 1}: UNKNOWN_SOURCE_SEGMENT')
                excerpts = {}
                break
            eid, quote = segments[selection.segment_id]
            if eid not in registry or quote not in registry[eid].excerpt:
                raise ValueError('원문 구간과 검색 원장 불일치')
            excerpts.setdefault(eid, quote)
        if excerpts:
            accepted.append(Finding(topic=item.topic, claim=item.claim, evidence_ids=list(excerpts), excerpts=excerpts))
        else:
            issues.append(f'조사 항목 {index + 1}: 검증 가능한 원문 선택 없음')
    return accepted, issues


def web_search(query):
    key = os.environ.get('TAVILY_API_KEY')
    if not key:
        raise ValueError('TAVILY_API_KEY required')
    request = Request('https://api.tavily.com/search',
                      data=json.dumps({'api_key': key, 'query': query, 'max_results': 5}).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=15) as response:
        raw = json.load(response)
    return [{'source_type': 'web', 'source': x['url'], 'source_title': x.get('title') or x['url'],
             'excerpt': x['content'], 'page': None, 'raw_document': x} for x in raw.get('results', []) if x.get('url') and x.get('content')]


def structured(model, schema, prompt, payload):
    from .input_limits import prepare_payload
    payload = prepare_payload(payload, schema.__name__)
    output = model.with_structured_output(schema).invoke([
        ('system', prompt + '\n설명과 보고서 본문은 한국어로 작성한다. 고유명사·URL·근거 ID·원문 발췌는 원본을 유지한다.'), ('user', json.dumps(payload, ensure_ascii=False, allow_nan=False))])
    return schema.model_validate(output.model_dump() if hasattr(output, 'model_dump') else output)


def collect(task, *, prompt, model_env, model=None, searchers=None):
    company, role = task['company_name'], task['agent']
    if searchers is None:
        searchers = [web_search]
        if role == 'market_scout':
            from rag.market_scout import rag_search
            searchers.append(rag_search)
        elif role == 'tech_brief':
            from .tech_brief import make_searchers
            searchers = make_searchers(company)
    previous = task.get('previous_result') or {}
    evidence = {x['evidence_id']: Evidence.model_validate(x) for x in previous.get('evidence', [])}
    raw_documents = deepcopy(previous.get('raw_documents', {}))
    errors = []
    missing = []
    queries = list(dict.fromkeys(q if company.casefold() in q.casefold() else f'{company} {q}' for q in task['queries']))
    for query in queries:
        for search in searchers:
            try:
                rows = search(query)
                for row in rows:
                    if '_missing_item' in row:
                        missing.append(row['_missing_item'])
                        continue
                    normalized = {key: row.get(key) for key in ('source_title', 'source', 'source_type', 'excerpt', 'page')}
                    normalized['provenance'] = row.get('provenance', 'unknown')
                    eid = role + ':' + hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
                    validated = Evidence(evidence_id=eid, **normalized)
                    # raw 산출물은 LLM이 작성하지 않고 검색기가 반환한 값을 직접 보존한다.
                    raw_data = deepcopy(row.get('raw_document', row))
                    json.dumps(raw_data, ensure_ascii=False, allow_nan=False)
                    record = raw_documents.get(eid)
                    if record is None:
                        raw_documents[eid] = {'data': raw_data, 'queries': [query]}
                    elif query not in record['queries']:
                        record['queries'].append(query)
                    evidence[eid] = validated
            except Exception as exc:
                # 원문 에러/stack trace/비밀키는 payload와 State에 넣지 않는다.
                errors.append(f'{getattr(search, "__name__", "search")}: {type(exc).__name__}')
    if not evidence:
        if errors:
            raise RuntimeError('All searches failed or returned no usable evidence')
        return Collection(company_name=company, agent=role, findings=[], evidence=[],
                          missing_items=list(dict.fromkeys(missing + ['직접 관련 근거 없음'])), queries_used=queries).model_dump()
    selected = model if model is not None else create_model_from_env(model_env_var=model_env, timeout=40, max_retries=0)
    segments, documents = source_segments(evidence)
    try:
        result = structured(selected, extraction_schema(list(segments)), prompt, {
            'company_name': company, 'request': task['request'], 'instruction': task['instruction'],
            'previous_findings': previous.get('findings', []),
            'documents': documents, 'search_errors': errors})
        findings, validation_issues = validated_findings(result.findings, evidence, segments)
        missing.extend(result.missing_items)
        missing.extend(validation_issues)
    except ValidationError as exc:
        # 잘못된 요약 응답 때문에 이미 검색한 raw 데이터까지 잃지 않는다.
        findings = []
        detail = safe_error_details(exc)
        missing.append('조사 요약 응답 형식 검증 실패: ' + json.dumps(detail, ensure_ascii=False))
        errors.append('generation: ValidationError')
    # 추가 조사나 요약 실패가 기존의 유효한 내용을 지우지 않도록 병합한다.
    retained = [Finding.model_validate(f) for f in previous.get('findings', [])]
    combined = {}
    for finding in retained + findings:
        key = (finding.topic, finding.claim, tuple(sorted(finding.evidence_ids)))
        combined[key] = finding
    findings = list(combined.values())
    if not findings:
        missing.append('원문으로 검증된 조사 요약 없음. 추가 조사/요약 필요')
    return Collection(company_name=company, agent=role, findings=findings,
                      evidence=list(evidence.values()), missing_items=list(dict.fromkeys(missing)),
                      queries_used=list(dict.fromkeys(previous.get('queries_used', []) + queries)),
                      search_errors=errors, raw_documents=raw_documents).model_dump()



def research_node(role, handler, **agent_kwargs):
    """공통 State의 조사 payload를 관점별로 교체한다. 파일로 저장하지 않는다."""
    import logging
    from datetime import datetime, timezone
    from .contracts import ResearchTask
    logger = logging.getLogger(__name__)
    def log(state, event, **details):
        logger.info(json.dumps({'trace_id': state['trace_id'], 'node': role, 'event': event,
                                'ts': datetime.now(timezone.utc).isoformat(), **details}, ensure_ascii=False))
    def node(state):
        attempt = state.get('node_attempts', {}).get(role, 0) + 1
        try:
            plan = ResearchTask.model_validate(state['research_tasks'][role])
            if plan.agent != role:
                raise ValueError('Assigned agent differs from node role')
            previous = state.get('perspectives', {}).get(role)
            if previous and previous['company_name'] != state['company_name']:
                raise ValueError('Previous research belongs to a different company')
            task = {**plan.model_dump(), 'company_name': state['company_name'],
                    'request': state['research_request'], 'previous_result': previous}
            if role == 'tech_brief':
                task['product_names'] = state.get('product_names', [])
            result = Collection.model_validate(handler(task, **agent_kwargs)).model_dump()
            if result['company_name'] != state['company_name'] or result['agent'] != role:
                raise ValueError('Research result identity mismatch')
            log(state, 'research_complete', findings_count=len(result['findings']), evidence_count=len(result['evidence']), partial=bool(result['missing_items'] or result['search_errors']))
            return {'perspectives': {role: result},
                    'node_status': {role: 'success' if result['findings'] and not result['search_errors'] and not result['missing_items'] else 'partial'},
                    'node_attempts': {role: attempt}, 'node_errors': {role: None}}
        except Exception as exc:
            log(state, 'research_error', error_type=type(exc).__name__, validation_details=safe_error_details(exc))
            return {'node_status': {role: 'error'}, 'node_attempts': {role: attempt},
                    'node_errors': {role: type(exc).__name__}}
    return node
