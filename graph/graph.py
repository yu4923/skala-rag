"""현재 조사 State와 근거 충분성에 따라 동적으로 분기하는 Supervisor Graph."""
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from importlib import import_module
from dotenv import dotenv_values, load_dotenv
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from agents.contracts import Collection, ROLES, ResearchDecision
from .supervisor_state import State, initial_state

ENV_PATH = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(ENV_PATH, override=False)


def setting(name, convert=str, default=None):
    value = os.environ.get(name, dotenv_values(ENV_PATH).get(name))
    if value is None or not str(value).strip():
        if default is None:
            raise ValueError(f'{name} 설정이 필요합니다.')
        value = default
    return convert(value)


ENABLE_LANGSMITH = setting('LANGSMITH_TRACING', default='false').lower() == 'true'
GRAPH_RECURSION_LIMIT = setting('GRAPH_RECURSION_LIMIT', int, 500)


def configure_logging(enabled):
    # 파일 handler를 만들지 않는다. 관측성은 콘솔 logger 또는 호출자 콜백으로 분리한다.
    for name in ('graph.graph', 'agents.collection'):
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.propagate = False
    os.environ['LANGSMITH_TRACING'] = str(enabled).lower()
    os.environ['LANGCHAIN_TRACING_V2'] = str(enabled).lower()
    if enabled:
        if not setting('LANGSMITH_API_KEY', default=''):
            raise ValueError('LANGSMITH_API_KEY 설정이 필요합니다.')
        os.environ['LANGSMITH_PROJECT'] = setting('LANGSMITH_PROJECT', default='RAG-Supervisor')


def make_initial_state(companies, request, **kwargs):
    company = companies if isinstance(companies, str) else (companies[0] if companies else '')
    return initial_state(company, research_request=request, **kwargs)


def build_graph(*, supervisor_handler=None, workers=None, event_logger=None):
    workers = workers or {}

    def log(state, event, **details):
        record = {'trace_id': state['trace_id'], 'node': 'supervisor', 'event': event,
                  'ts': datetime.now(timezone.utc).isoformat(), **details}
        if event_logger is not None:
            event_logger(record)
        else:
            logging.getLogger(__name__).info(json.dumps(record, ensure_ascii=False))

    def stop(state, code, reason, *, step_count=None):
        result = {'action': 'blocked', 'reason': reason, 'evidence_sufficient': False}
        log(state, 'decision', decision='blocked', reason=reason)
        return {'action': 'blocked', 'status': 'blocked', 'evidence_sufficient': False,
                'verdict': result, 'research_tasks': {}, 'last_error': code,
                'step_count': state['step_count'] if step_count is None else step_count}

    def supervisor(state):
        # 종료 상한은 비정상 루프 방지용이며 충분성을 대신하지 않는다.
        if state['step_count'] >= state['max_steps']:
            return stop(state, 'STEP_LIMIT', 'Supervisor 실행 상한 도달. 근거 충분성을 인정하지 않음.')
        step = state['step_count'] + 1
        try:
            research = {}
            for role, perspective in state['perspectives'].items():
                if role not in ROLES or perspective['company_name'] != state['company_name']:
                    raise ValueError('관점 결과의 기업 또는 Agent 불일치')
                result = Collection.model_validate(perspective)
                if result.company_name != state['company_name'] or result.agent != role:
                    raise ValueError('전체 수집 결과의 기업 또는 Agent 불일치')
                research[role] = result.model_dump()
            available = [role for role in ROLES if state['node_attempts'].get(role, 0) < state['max_retries'] + 1]
            payload = {'company_name': state['company_name'], 'request': state['research_request'],
                       'perspectives': state['perspectives'], 'research': research,
                       'node_status': state['node_status'], 'node_attempts': state['node_attempts'],
                       'node_errors': state['node_errors'], 'available_agents': available,
                       'remaining_steps': state['max_steps'] - state['step_count']}
            if supervisor_handler:
                output = supervisor_handler(payload)
            else:
                from agents.supervisor import run
                output = run(payload)
            decision = ResearchDecision.model_validate(output)
            actual = {r for r, data in research.items() if data['findings'] and data['evidence']}
            if set(decision.collected_perspectives) != actual:
                raise ValueError('수집된 관점 선언과 실제 근거 불일치')
            if decision.action == 'finish':
                if not decision.evidence_sufficient or actual != set(ROLES) or any(state['node_status'].get(r) == 'error' for r in ROLES):
                    raise ValueError('finish에는 현재 세 관점의 직접 근거와 Supervisor 충분성 판정이 필요함')
            if decision.action == 'collect':
                if any(t.agent not in available for t in decision.tasks):
                    return stop(state, 'ATTEMPT_LIMIT', '선택된 Agent의 재시도 상한 도달. 충분성 미확인.', step_count=step)
            log(state, 'decision', decision=decision.action, reason=decision.reason, plan=decision.model_dump())
            return {'action': decision.action,
                    'status': 'finished' if decision.action == 'finish' else 'blocked' if decision.action == 'blocked' else 'running',
                    'evidence_sufficient': decision.evidence_sufficient if decision.action == 'finish' else False,
                    'verdict': ({'action': decision.action, 'reason': decision.reason,
                                 'evidence_sufficient': decision.evidence_sufficient,
                                 'missing_items': decision.missing_items} if decision.action != 'collect' else None),
                    'step_count': step,
                    'research_tasks': {t.agent: t.model_dump() for t in decision.tasks},
                    'last_error': 'EVIDENCE_INSUFFICIENT' if decision.action == 'blocked' else None}
        except Exception as exc:
            # 외부 API 오류 본문/인증 정보는 저장하지 않고 오류 유형만 유지한다.
            return stop(state, type(exc).__name__, f'Supervisor 실행 또는 계약 검증 실패: {type(exc).__name__}', step_count=step)

    def route(state):
        # 고정 순서/스텝 수 대신 현재 State의 판단과 선택 작업으로 분기한다.
        if state['action'] != 'collect':
            return END
        return [Send(role, dict(state)) for role in state['research_tasks']]

    builder = StateGraph(State)
    builder.add_node('supervisor', supervisor)
    for role in ROLES:
        worker = workers.get(role)
        if worker is None:
            worker = import_module(f'agents.{role}').as_node()
        builder.add_node(role, worker)
        builder.add_edge(role, 'supervisor')
    builder.add_edge(START, 'supervisor')
    builder.add_conditional_edges('supervisor', route)
    # State는 실행 메모리에만 유지한다. SQLite/checkpointer는 사용하지 않는다.
    return builder.compile()
