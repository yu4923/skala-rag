"""내용 충분성은 모델이 판단하고 실행 가능한 선택지는 State 기반 스키마로 제한한다."""
from typing import Literal
from pydantic import Field, create_model
from .collection import structured
from .contracts import ResearchDecision, ResearchTask
from .evaluation_support import create_model_from_env
from prompts.supervisor_prompt import SUPERVISOR_PROMPT


def decision_schema(allowed_actions, available_agents):
    task_schema = ResearchTask
    if available_agents:
        task_schema = create_model('AvailableResearchTask', __base__=ResearchTask,
                                   agent=(Literal[tuple(available_agents)], ...))
    return create_model('ResearchDecision', __base__=ResearchDecision,
                        action=(Literal[tuple(allowed_actions)], ...),
                        tasks=(list[task_schema], Field(default_factory=list, max_length=3 if 'collect' in allowed_actions else 0)))


def run(payload, *, model=None):
    selected = model if model is not None else create_model_from_env(model_env_var='SUPERVISOR_MODEL', timeout=45, max_retries=0)
    schema = decision_schema(payload['allowed_actions'], payload['available_agents'])
    return structured(selected, schema, SUPERVISOR_PROMPT, payload).model_dump()
