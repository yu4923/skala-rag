"""보고서 검토: 인용 연결은 코드, 의미/중립성/편향은 LLM으로 검토한다."""
from .collection import structured
from .contracts import ReportReview, Contract
from pydantic import Field, create_model
import re
from .evaluation_support import create_model_from_env
from .report_generator import evidence_registry, validate_citations, CITATION_PATTERN
from prompts.reviewer_prompt import REVIEWER_PROMPT


class ReviewIssue(Contract):
    report_excerpt: str = Field(min_length=1)
    problem: str = Field(min_length=1)


class DetailedReviewItem(Contract):
    passed: bool
    reason: str = Field(min_length=1)
    issues: list[ReviewIssue]


# 모델 입력 예산의 기존 검토 역할을 유지한다.
Assessment = create_model('ReportReview', __base__=Contract,
                          groundedness=(DetailedReviewItem, ...), neutrality=(DetailedReviewItem, ...),
                          bias_control=(DetailedReviewItem, ...), required_actions=(list[str], ...))


def validated_review(assessment, report):
    body = '\n'.join(v for v in report['draft'].values() if isinstance(v, str))
    normalize = lambda s: re.sub(r'\s+', '', CITATION_PATTERN.sub('', s))
    searchable = normalize(body)
    result = {}
    for field in ('groundedness', 'neutrality', 'bias_control'):
        item = getattr(assessment, field)
        issues = []
        for issue in item.issues:
            if not normalize(issue.report_excerpt) or normalize(issue.report_excerpt) not in searchable:
                raise ValueError('평가 지적의 발췌가 실제 보고서에 없음')
            issues.append(f'「{issue.report_excerpt}」: {issue.problem}')
        if not item.passed and not issues:
            raise ValueError('미통과 평가는 실제 보고서의 문제 문장을 제시해야 함')
        result[field] = {'passed': item.passed, 'reason': item.reason, 'issues': issues}
    result['required_actions'] = assessment.required_actions
    return ReportReview.model_validate(result).model_dump()


def run(payload, *, model=None):
    try:
        validate_citations(payload['report']['draft'], evidence_registry(payload['research']))
    except (ValueError, KeyError):
        return ReportReview.model_validate({
            'groundedness': {'passed': False, 'reason': '본문 인용과 수집 출처 연결 실패', 'issues': ['존재하지 않거나 불일치하는 근거 ID']},
            'neutrality': {'passed': False, 'reason': '검토 미수행', 'issues': ['인용 오류 수정 후 검토 필요']},
            'bias_control': {'passed': False, 'reason': '검토 미수행', 'issues': ['인용 오류 수정 후 검토 필요']},
            'required_actions': ['수집 근거와 본문 인용을 일치시키기']}).model_dump()
    payload = dict(payload)
    registry = evidence_registry(payload['research'])
    payload['cited_sources'] = [{key: registry[eid].get(key) for key in ('evidence_id', 'source_title', 'source', 'source_type', 'page', 'provenance')}
                               for eid in payload['report']['draft']['used_evidence_ids']]
    selected = model if model is not None else create_model_from_env(model_env_var='REVIEWER_MODEL', timeout=60, max_retries=0)
    for attempt in range(2):
        assessment = structured(selected, Assessment, REVIEWER_PROMPT, payload)
        try:
            return validated_review(assessment, payload['report'])
        except ValueError as exc:
            if attempt:
                raise
            payload['assessment_format_error'] = str(exc)
            payload['previous_assessment'] = assessment.model_dump()
            # 판정 통과로 바꾸지 않고, 실제 본문의 발췌에 연결된 평가를 다시 요청한다.
