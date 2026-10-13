"""founder_insight: 점수 없이 자료와 원문 근거를 수집한다."""
from .collection import collect, research_node
from prompts.founder_insight_prompt import FOUNDER_INSIGHT_PROMPT


def run(task, **kwargs):
    return collect(task, prompt=FOUNDER_INSIGHT_PROMPT, model_env='FOUNDER_MODEL', **kwargs)


def as_node(**kwargs):
    return research_node('founder_insight', run, **kwargs)
