"""보고서 검토: 인용 연결은 코드, 의미/중립성/편향은 LLM으로 검토한다."""
from .collection import structured
from .contracts import ReportReview
from .evaluation_support import create_model_from_env
from .report_generator import evidence_registry, validate_citations
from prompts.reviewer_prompt import REVIEWER_PROMPT


def run(payload, *, model=None):
    try:
        validate_citations(payload['report']['draft'], evidence_registry(payload['research']))
    except (ValueError, KeyError):
        return ReportReview.model_validate({
            'groundedness': {'passed': False, 'reason': '본문 인용과 수집 출처 연결 실패', 'issues': ['존재하지 않거나 불일치하는 근거 ID']},
            'neutrality': {'passed': False, 'reason': '검토 미수행', 'issues': ['인용 오류 수정 후 검토 필요']},
            'bias_control': {'passed': False, 'reason': '검토 미수행', 'issues': ['인용 오류 수정 후 검토 필요']},
            'required_actions': ['수집 근거와 본문 인용을 일치시키기']}).model_dump()
    selected = model if model is not None else create_model_from_env(model_env_var='REVIEWER_MODEL', timeout=60, max_retries=0)
    return structured(selected, ReportReview, REVIEWER_PROMPT, payload).model_dump()
