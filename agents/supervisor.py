"""현재 근거 충분성에 따라 수집/완료/보류를 결정한다."""
from .collection import structured
from .contracts import ResearchDecision
from .evaluation_support import create_model_from_env
from prompts.supervisor_prompt import SUPERVISOR_PROMPT


def run(payload, *, model=None):
    selected = model if model is not None else create_model_from_env(model_env_var='SUPERVISOR_MODEL', timeout=45, max_retries=0)
    return structured(selected, ResearchDecision, SUPERVISOR_PROMPT, payload).model_dump()
