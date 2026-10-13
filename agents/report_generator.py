"""중립적 기업 분석 보고서. 제목/작성일/REFERENCE는 코드가 확정한다."""
import re
from copy import deepcopy
from pydantic import create_model, Field
from typing import Literal
from .collection import structured
from .contracts import ReportDraft, ReportReview, ReportSection, ReportParagraph
from .evaluation_support import create_model_from_env
from prompts.report_generator_prompt import REPORT_GENERATOR_PROMPT, REPORT_SECTION_PROMPT, REPORT_SUMMARY_PROMPT, REPORT_EDIT_PROMPT

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
    refs, seen_locations = [], set()
    for eid in dict.fromkeys(draft.used_evidence_ids):
        e = registry[eid]
        key = (e['source'], e.get('page'))
        if key in seen_locations:
            continue
        seen_locations.add(key)
        location = f" / p. {e['page']}" if e.get('page') else ''
        refs.append(f"- {e['source_title']}{location} / {e['source']}")
    sections.append(('## REFERENCE', '\n'.join(refs)))
    return f'# {company} 기업 분석 보고서\n\n작성일: {report_date}\n\n' + '\n\n'.join(f'{heading}\n\n{display_citations(body, registry)}' for heading, body in sections)


SECTION_ROLES = {
    'business_overview': ('founder_insight',),
    'market_analysis': ('market_scout',),
    'product_technology_and_team': ('tech_brief',),
    'limitations': ('founder_insight', 'market_scout', 'tech_brief'),
}


def section_payload(payload, field):
    """원본 State는 유지하고 해당 장의 원문/요약만 모델에 전달한다."""
    data = {k: deepcopy(v) for k, v in payload.items() if k not in ('research', 'report')}
    research = {}
    for role in SECTION_ROLES[field]:
        result = deepcopy(payload['research'][role])
        if field == 'limitations':
            # 원문 중복 전송을 줄이고 실제 요약 연결 원문을 우선한다.
            referenced = {eid for f in result['findings'] for eid in f['evidence_ids']}
            result['evidence'] = [e for e in result['evidence'] if e['evidence_id'] in referenced]
            result.pop('raw_documents', None)
        research[role] = result
    data['research'] = research
    data['section'] = field
    data['source_inventory'] = [{'agent': role, 'topic': f['topic'], 'evidence_ids': f['evidence_ids']}
                                for role, result in research.items() for f in result['findings']]
    previous = payload.get('report')
    if previous:
        data['previous_section'] = previous.get('draft', {}).get(field, '')
    return data


def section_schema(ids):
    paragraph = create_model('CitedParagraph', __base__=ReportParagraph,
                             evidence_ids=(list[Literal[tuple(ids)]], ...))
    return create_model('ReportSection', __base__=ReportSection,
                        paragraphs=(list[paragraph], Field(min_length=1)))


def render_section(section, registry=None):
    parts = []
    for paragraph in section.paragraphs:
        # 인용은 구조화된 원장 ID로 코드가 붙인다. 모델이 ID를 본문에 재작성하지 않는다.
        if CITATION_PATTERN.search(paragraph.text):
            raise ValueError('구조화된 문단의 본문에는 근거 표기를 직접 넣을 수 없습니다.')
        heading = f'### {paragraph.heading}\n\n' if paragraph.heading else ''
        marks = ''.join(f'[근거:{eid}]' for eid in dict.fromkeys(paragraph.evidence_ids))
        text = paragraph.text
        if registry is not None:
            originals = ' '.join(registry[eid]['excerpt'] for eid in paragraph.evidence_ids).casefold()
            # 출처에 없는 영문 약어 풀이는 새 사실이므로 본문에 추가하지 않는다.
            text = re.sub(r'\b([A-Z][A-Z0-9]{1,7})\s*\(([A-Za-z][A-Za-z -]{3,80})\)',
                          lambda m: m.group(0) if m.group(2).casefold() in originals else m.group(1), text)
        parts.append(heading + text + marks)
    return '\n\n'.join(parts)


def run(payload, *, model=None):
    selected = model if model is not None else create_model_from_env(model_env_var='REPORT_MODEL', timeout=90, max_retries=0)
    registry = evidence_registry(payload['research'])
    chapters, used = {}, []
    for field in SECTION_ROLES:
        data = section_payload(payload, field)
        ids = list(evidence_registry(data['research']))
        schema = section_schema(ids)
        part = structured(selected, schema, REPORT_SECTION_PROMPT, data)
        cited = {eid for paragraph in part.paragraphs for eid in paragraph.evidence_ids}
        editing = deepcopy(data)
        for result in editing['research'].values():
            result.pop('raw_documents', None)
            result['evidence'] = [e for e in result['evidence'] if e['evidence_id'] in cited]
        editing['paragraphs'] = part.model_dump()['paragraphs']
        part = structured(selected, schema, REPORT_EDIT_PROMPT, editing)
        chapters[field] = render_section(part, registry)
        used.extend(eid for para in part.paragraphs for eid in para.evidence_ids)
    # SUMMARY는 작성된 본문을 요약한다. 전체 조사 raw를 다시 전송하지 않는다.
    summary = structured(selected, section_schema(list(dict.fromkeys(used))), REPORT_SUMMARY_PROMPT,
                         {'company_name': payload['company_name'], 'section': 'summary',
                          'chapters': chapters, 'report_review': payload.get('report_review')})
    summary_ids = [eid for para in summary.paragraphs for eid in para.evidence_ids]
    draft = ReportDraft(summary=render_section(summary, registry), **chapters, used_evidence_ids=list(dict.fromkeys(used + summary_ids)))
    draft = normalize_citations(draft, registry)
    return {'company_name': payload['company_name'], 'draft': draft.model_dump(),
            'markdown': render_markdown(draft, payload['company_name'], payload['report_date'], registry)}


BRIEF_CHAR_LIMIT = 1400


def render_deferred_company(state, index=1):
    """짧은 기업별 보류 요약. 최종 보고서 승인을 받은 결과로 표시하지 않는다."""
    def short(text, limit):
        text = ' '.join(str(text).split())
        return text if len(text) <= limit else text[:limit - 1] + '…'
    labels = {'founder_insight': '창업자·기업(팀)', 'market_scout': '시장성', 'tech_brief': '기업 기술'}
    header = f"## {index}. {state['company_name']}\n\n작성일: {state['report_date']}"
    summaries, references, gaps = [], [], []
    for role, label in labels.items():
        result = state['perspectives'].get(role)
        if not result or not result['findings']:
            summaries.append(f'- {label}: 검증된 조사 요약 미확보')
            continue
        registry = {e['evidence_id']: e for e in result['evidence']}
        finding = result['findings'][0]
        sources = [registry[eid] for eid in finding['evidence_ids'] if eid in registry]
        # 아주 긴 URL도 임의로 잘라 다른 주소로 표시하지 않는다.
        usable = next((e for e in sources if len(source_location(e)) <= 160), None)
        mark = ''
        if usable:
            location = source_location(usable)
            if location not in references:
                references.append(location)
            mark = f' [출처 {references.index(location) + 1}]'
        summaries.append(f"- {label}: {short(finding['claim'], 100)}{mark}")
        gaps.extend(f'{label}: {short(x, 90)}' for x in result['missing_items'][:1])
    attempts = state['node_attempts'].get('report_generator', 0)
    stage = f'보고서 초안 작성 {attempts}회, 승인 미완료' if attempts else '정식 보고서 작성 전 조사 단계에서 보류'
    issues = [f"- 처리 단계: {stage}", '- 보류 사유: ' + short((state.get('verdict') or {}).get('reason', '근거 또는 검토 조건 미충족'), 200)]
    if state.get('report_review'):
        review = ReportReview.model_validate(state['report_review'])
        for name, item in (('근거 연결', review.groundedness), ('중립성', review.neutrality), ('편향 통제', review.bias_control)):
            if not item.passed or item.issues:
                detail = '; '.join(item.issues) or item.reason
                issues.append(f'- {name}: {short(detail, 120)}')
    elif attempts:
        issues.append('- 검토: 현재 초안의 승인된 검토 결과 없음')
    if gaps:
        issues.append('- 남은 확인사항: ' + short('; '.join(gaps), 220))
    errors = [f'{node}: {code}' for node, code in state['node_errors'].items() if code]
    if errors:
        issues.append('- 실행 한계: ' + short('; '.join(errors), 140))
    refs = '\n'.join(f'{i}. {location}' for i, location in enumerate(references, 1)) or '본문에 연결 가능한 짧은 출처 없음. 원본 출처는 조사 State에 보존됨.'
    page = header + '\n\n### 조사 요약\n\n' + '\n'.join(summaries) + '\n\n### 작성·승인 한계\n\n' + '\n'.join(issues) + '\n\n### 출처\n\n' + refs
    # 필수 요약/보류 사유를 유지하면서 부수 설명을 줄여 한 기업의 분량을 제한한다.
    while len(page) > BRIEF_CHAR_LIMIT and len(issues) > 2:
        issues.pop()
        page = header + '\n\n### 조사 요약\n\n' + '\n'.join(summaries) + '\n\n### 작성·승인 한계\n\n' + '\n'.join(issues) + '\n\n### 출처\n\n' + refs
    return page


def render_deferred_report(states):
    date = states[0]['report_date'] if states else ''
    introduction = f'# 기업별 조사 보류 보고서\n\n작성일: {date}\n\n## SUMMARY\n\n모든 후보에서 승인된 분석 보고서를 얻지 못했습니다. 다음은 각 기업의 짧은 조사 요약과 작성·승인 한계입니다.'
    return introduction + '\n\n---\n\n' + '\n\n---\n\n'.join(render_deferred_company(state, i) for i, state in enumerate(states, 1))
