"""State는 보존하고 모델 입력만 역할별로 분리·제한한다."""
from copy import deepcopy
import json


def prepare_payload(payload, schema_name, max_bytes=64000):
    data = deepcopy(payload)
    if 'research' in data:
        data.pop('perspectives', None)  # 동일 결과 중복 전송 제거
    if isinstance(data.get('report'), dict):
        data['report'].pop('markdown', None)  # draft와 동일 본문 중복 제거
    if schema_name == 'ResearchDecision':
        for result in data.get('research', {}).values():
            result.pop('raw_documents', None)
            for finding in result.get('findings', []):
                finding.pop('excerpts', None)
            for evidence in result.get('evidence', []):
                evidence.pop('excerpt', None)
        if data.get('report'):
            report = data['report']
            data['report'] = {'company_name': report.get('company_name'),
                              'summary': report.get('draft', {}).get('summary', '')}
    elif schema_name == 'ReportReview':
        for result in data.get('research', {}).values():
            result.pop('raw_documents', None)
    if schema_name in ('ReportDraft', 'ReportSection', 'ReportReview'):
        cited = set((data.get('report') or {}).get('draft', {}).get('used_evidence_ids', []))
        for result in data.get('research', {}).values():
            needed = cited | {eid for f in result.get('findings', []) for eid in f.get('evidence_ids', [])}
            evidence = result.get('evidence', [])
            selected = [e for e in evidence if e['evidence_id'] in needed]
            selected += [e for e in evidence if e not in selected][:max(0, 12 - len(selected))]
            result['evidence'] = selected
            ids = {e['evidence_id'] for e in selected}
            if 'raw_documents' in result:
                result['raw_documents'] = {eid: raw for eid, raw in result['raw_documents'].items() if eid in ids}
            result['queries_used'] = result.get('queries_used', [])[-10:]
    # 보고서 작성만 raw 데이터를 사용한다. State의 원문을 수정하지 않는다.
    def encode(value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(encode(data).encode('utf-8')) <= max_bytes:
        return data
    referenced = set((data.get('report') or {}).get('draft', {}).get('used_evidence_ids', []))
    referenced.update(eid for result in data.get('research', {}).values() for finding in result.get('findings', []) for eid in finding.get('evidence_ids', []))
    def shrink(value, limit, key=''):
        if isinstance(value, str):
            # 출처/식별자는 잘라서 새 값을 만들지 않는다.
            if key in ('source', 'source_url', 'url', 'evidence_id', 'company_name', 'agent'):
                return value
            if schema_name == 'ReportSection' and key in ('claim', 'topic', 'body'):
                return value
            return value[:limit]
        if isinstance(value, list):
            selected = value
            if key in ('evidence', 'documents'):
                selected = [v for v in value if isinstance(v, dict) and v.get('evidence_id') in referenced]
                selected += [v for v in value if v not in selected][:20]
            return [shrink(v, limit) for v in selected]
        if isinstance(value, dict):
            return {k: shrink(v, limit, k) for k, v in value.items()}
        return value
    for limit in (2000, 1000, 500, 200, 80):
        clipped = shrink(data, limit)
        # Reviewer가 보고서 일부만 읽고 승인하지 않도록 본문 전체는 보존한다.
        if schema_name in ('ReportReview', 'ReportSection') and data.get('report'):
            clipped['report'] = data['report']
        clipped['_input_limits'] = {'excerpt_preview': True, 'message': '검색 원문 일부가 길이 제한으로 축약됨. 보이지 않는 내용의 사실성을 추정하지 말 것.'}
        if len(encode(clipped).encode('utf-8')) <= max_bytes:
            return clipped
    raise ValueError('모델 입력 예산 초과: 문서 선택 또는 더 작은 보고서 단위가 필요합니다.')
