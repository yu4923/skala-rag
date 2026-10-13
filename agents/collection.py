"""검색 원장을 보존하는 공통 자료 수집기. 평가 점수는 생성하지 않는다."""
import hashlib
import json
import os
from urllib.request import Request, urlopen
from .contracts import Collection, Evidence, Finding, Contract
from .evaluation_support import create_model_from_env
from pydantic import Field


class FindingsOutput(Contract):
    findings: list[Finding]
    missing_items: list[str] = Field(default_factory=list)


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
             'excerpt': x['content'], 'page': None} for x in raw.get('results', []) if x.get('url') and x.get('content')]


def structured(model, schema, prompt, payload):
    output = model.with_structured_output(schema).invoke([
        ('system', prompt), ('user', json.dumps(payload, ensure_ascii=False, allow_nan=False))])
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
                    evidence[eid] = Evidence(evidence_id=eid, **normalized)
            except Exception as exc:
                # 원문 에러/stack trace/비밀키는 payload와 State에 넣지 않는다.
                errors.append(f'{getattr(search, "__name__", "search")}: {type(exc).__name__}')
    if not evidence:
        if errors:
            raise RuntimeError('All searches failed or returned no usable evidence')
        return Collection(company_name=company, agent=role, findings=[], evidence=[],
                          missing_items=list(dict.fromkeys(missing + ['직접 관련 근거 없음'])), queries_used=queries).model_dump()
    selected = model if model is not None else create_model_from_env(model_env_var=model_env, timeout=40, max_retries=0)
    result = structured(selected, FindingsOutput, prompt, {
        'company_name': company, 'request': task['request'], 'instruction': task['instruction'],
        'previous_findings': previous.get('findings', []),
        'documents': [e.model_dump() for e in evidence.values()], 'search_errors': errors})
    # 새 생성 결과는 모든 finding이 검색 원문 발췌까지 연결되어야 한다.
    # 이전 저장 결과의 excerpts 부재는 재조사/재개 호환을 위해 계약 모델에서 허용한다.
    if any(set(f.excerpts) != set(f.evidence_ids) for f in result.findings):
        raise ValueError('각 조사 내용에 연결 근거별 원문 발췌가 필요합니다.')
    return Collection(company_name=company, agent=role, findings=result.findings,
                      evidence=list(evidence.values()), missing_items=list(dict.fromkeys(missing + result.missing_items)),
                      queries_used=list(dict.fromkeys(previous.get('queries_used', []) + queries)), search_errors=errors).model_dump()


def research_node(role, handler, **agent_kwargs):
    """세 Agent 공통 ResearchState -> 자기 관점 변경분 어댑터.

    실행 그래프는 연결하지 않는다. 호출자가 research_tasks로 질의를 배정한다.
    원문 payload와 관측성은 외부 저장, 공유 State에는 유한한 조사 요약만 반환한다.
    """
    from graph.artifacts import ArtifactStore
    from .contracts import ResearchTask
    def node(state):
        trace_id = state['trace_id']
        storage = ArtifactStore(trace_id)
        context = {'run_id': trace_id, 'trace_id': trace_id}
        attempt = state.get('node_attempts', {}).get(role, 0) + 1
        try:
            plan = ResearchTask.model_validate(state['research_tasks'][role])
            if plan.agent != role:
                raise ValueError('Assigned agent differs from node role')
            previous = state.get('perspectives', {}).get(role)
            previous_data = None
            if previous and previous['company_name'] == state['company_name']:
                previous_data = storage.get(previous['result_ref'])
            task = {**plan.model_dump(), 'company_name': state['company_name'],
                    'request': state['research_request'], 'previous_result': previous_data}
            if role == 'tech_brief':
                task['product_names'] = state.get('product_names', [])
            result = Collection.model_validate(handler(task, **agent_kwargs)).model_dump()
            if result['company_name'] != state['company_name'] or result['agent'] != role:
                raise ValueError('Research result identity mismatch')
            ref = storage.put(result)
            def source(e):
                return e['source'] if e['source_type'] == 'web' else (f"{e['source_title']}, p. {e['page']}" if e.get('page') else e['source'])
            perspective = {'company_name': result['company_name'], 'view': role,
                           'findings': [f['claim'][:300] for f in result['findings'][:6]],
                           'sources': list(dict.fromkeys(source(e) for e in result['evidence']))[:12],
                           'missing_items': [m[:300] for m in result['missing_items'][:6]], 'result_ref': ref}
            storage.log(context, 'research_complete', node=role, output_ref=ref)
            return {'perspectives': {role: perspective}, 'node_status': {role: 'success' if result['findings'] and not result['search_errors'] and not result['missing_items'] else 'partial'},
                    'node_attempts': {role: attempt}, 'node_errors': {role: None}}
        except Exception as exc:
            storage.log(context, 'research_error', node=role, error_type=type(exc).__name__)
            return {'node_status': {role: 'error'}, 'node_attempts': {role: attempt},
                    'node_errors': {role: type(exc).__name__}}
    return node
