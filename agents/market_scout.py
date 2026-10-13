"""market_scout: 점수 없이 자료와 원문 근거를 수집한다."""
from .collection import collect, research_node
from prompts.market_scout_prompt import MARKET_SCOUT_PROMPT


def run(task, **kwargs):
    return collect(task, prompt=MARKET_SCOUT_PROMPT, model_env='MARKET_MODEL', **kwargs)


def as_node(**kwargs):
    return research_node('market_scout', run, **kwargs)
