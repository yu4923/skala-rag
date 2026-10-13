"""중립적 기업 분석 보고서. 제목/작성일/REFERENCE는 코드가 확정한다."""
import re
from .collection import structured
from .contracts import ReportDraft, ReportReview
from .evaluation_support import create_model_from_env
from prompts.report_generator_prompt import REPORT_GENERATOR_PROMPT

FIELDS = ('summary', 'business_overview', 'market_analysis', 'product_technology_and_team', 'limitations')


def evidence_registry(research):
    return {item['evidence_id']: item for result in research.values() for item in result['evidence']}


CITATION_PATTERN = re.compile(r'\[근거:([^\]\n]+)\]')


def normalize_citations(draft, registry):
    """알려진 표기 오류만 교정하고 실제 본문 인용으로 사용 목록을 구성한다."""
    draft = ReportDraft.model_validate(draft).model_copy(deep=True)
    used = []
    def normalize(match):
        eid = match.group(1).strip()
        if eid not in registry and eid.startswith('evidence_id:'):
            candidate = eid.removeprefix('evidence_id:')
            if candidate in registry:
                eid = candidate
        if eid not in registry:
            raise ValueError('본문이 실제 수집 자료에 없는 근거 ID를 인용했습니다.')
        if eid not in used:
            used.append(eid)
        return f'[근거:{eid}]'
    for field in FIELDS:
        text = getattr(draft, field)
        # 일부 모델이 생성하는 [evidence_id:실제값] 표기도 동일 원장으로 검증한다.
        text = re.sub(r'\[evidence_id:([^\]\n]+)\]', normalize, text)
        setattr(draft, field, CITATION_PATTERN.sub(normalize, text))
    if not used:
        raise ValueError('본문에 수집 근거를 연결하는 인용이 없습니다.')
    draft.used_evidence_ids = used
    return validate_citations(draft, registry)


def source_location(evidence):
    """내부 ID 대신 실제 URL 또는 PDF 파일명과 원본 페이지를 표시한다."""
    if evidence['source_type'] == 'web':
        return evidence['source']
    title = evidence['source_title']
    if evidence.get('page') is not None:
        return f"{title}, p. {evidence['page']}"
    return f"{title} / {evidence['source']}"


def display_citations(text, registry):
    return CITATION_PATTERN.sub(lambda m: f"[출처: {source_location(registry[m.group(1)])}]", text)


def validate_citations(draft, registry):
    draft = ReportDraft.model_validate(draft)
    cited = set(re.findall(r'\[근거:([^\]\n]+)\]', '\n'.join(getattr(draft, field) for field in FIELDS)))
    if not cited or cited != set(draft.used_evidence_ids) or cited - registry.keys():
        raise ValueError('Report citations must exactly match collected evidence IDs')
    return draft


def render_markdown(draft, company, report_date, registry):
    draft = validate_citations(draft, registry)
    sections = [('## SUMMARY', draft.summary), ('## 1. 사업 개요', draft.business_overview),
                ('## 2. 시장 및 고객', draft.market_analysis),
                ('## 3. 제품·기술 및 팀', draft.product_technology_and_team),
                ('## 4. 자료의 한계 및 추가 확인사항', draft.limitations)]
    refs = []
    for eid in dict.fromkeys(draft.used_evidence_ids):
        e = registry[eid]
        location = f" / p. {e['page']}" if e.get('page') else ''
        refs.append(f"- {e['source_title']}{location} / {e['source']} / 출처 성격: {e.get('provenance', 'unknown')}")
    sections.append(('## REFERENCE', '\n'.join(refs)))
    return f'# {company} 기업 분석 보고서\n\n작성일: {report_date}\n\n' + '\n\n'.join(f'{heading}\n\n{display_citations(body, registry)}' for heading, body in sections)


def run(payload, *, model=None):
    selected = model if model is not None else create_model_from_env(model_env_var='REPORT_MODEL', timeout=60, max_retries=0)
    draft = structured(selected, ReportDraft, REPORT_GENERATOR_PROMPT, payload)
    registry = evidence_registry(payload['research'])
    draft = normalize_citations(draft, registry)
    return {'company_name': payload['company_name'], 'draft': draft.model_dump(),
            'markdown': render_markdown(draft, payload['company_name'], payload['report_date'], registry)}



def render_deferred_report(states):
    """모든 후보가 미승인일 때 State의 사실과 보류 사유를 짧게 출력한다."""
    date = states[0]['report_date'] if states else ''
    parts = ['# 기업 조사 보류 보고서', f'작성일: {date}', '## SUMMARY',
             '승인된 분석 보고서가 없습니다. 아래는 수집된 조사 내용과 보류 사유이며, 검토를 통과한 분석 결과가 아닙니다.']
    references = []
    labels = {'founder_insight': '창업자·팀', 'market_scout': '시장·고객', 'tech_brief': '제품·기술'}
    for index, state in enumerate(states, 1):
        parts.append(f"## {index}. {state['company_name']}")
        lines = []
        for role, label in labels.items():
            result = state['perspectives'].get(role)
            if not result or not result['findings']:
                lines.append(f'- {label}: 관련 자료 확보 미완료')
                continue
            registry = {e['evidence_id']: e for e in result['evidence']}
            for finding in result['findings'][:2]:
                sources = [registry[eid] for eid in finding['evidence_ids'] if eid in registry]
                locations = list(dict.fromkeys(source_location(e) for e in sources))
                lines.append(f"- {label}: {finding['claim']} [출처: {'; '.join(locations)}]")
                references.extend(f"- {e['source_title']} / {source_location(e)}" for e in sources)
        parts.append('\n'.join(lines))
        parts.append('### 보류 사유\n\n' + (state.get('verdict') or {}).get('reason', '근거 또는 검토 조건 미충족'))
        if state.get('report_review'):
            verdict = ReportReview.model_validate(state['report_review'])
            for name, item in (('Groundedness', verdict.groundedness), ('중립성', verdict.neutrality), ('편향 통제', verdict.bias_control)):
                if not item.passed or item.issues:
                    parts.append(f"- {name}: {item.reason} {'; '.join(item.issues)}")
        errors = [f'{node}: {code}' for node, code in state['node_errors'].items() if code]
        if errors:
            parts.append('실행 오류: ' + '; '.join(errors))
    parts.extend(['## REFERENCE', '\n'.join(dict.fromkeys(references)) or '인용 가능한 자료 없음'])
    return '\n\n'.join(parts)
