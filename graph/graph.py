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


def build_graph(*, supervisor_handler=None, workers=None, report_handler=None, reviewer_handler=None, event_logger=None):
    from agents.contracts import ReportReview
    from agents.report_generator import evidence_registry, normalize_citations, render_markdown
    workers = workers or {}

    def log(state, event, node='supervisor', **details):
        record = {'trace_id': state['trace_id'], 'node': node, 'event': event,
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
        from .policy import inspect_state, validate_plan, PlanError
        if state['step_count'] >= state['max_steps']:
            return stop(state, 'STEP_LIMIT', '전체 실행 안전 상한 도달. 미승인 결과는 승인하지 않음.')
        step = state['step_count'] + 1
        try:
            policy = inspect_state(state)
            report = state['report'] if policy['report_current'] else None
            review = state['report_review'] if policy['report_current'] else None
            if policy['approved']:
                # 현재 근거/보고서 버전의 검토 통과가 확인되면 첫 승인 결과를 확정한다.
                decision = ResearchDecision(action='finish', reason='현재 보고서가 세 검토 기준을 통과했습니다.',
                                            evidence_sufficient=True, collected_perspectives=policy['covered'],
                                            missing_items=[], tasks=[])
            elif policy['allowed_actions'] == ['blocked']:
                missing = [r for r in ROLES if r not in policy['covered']]
                counts = ', '.join(f"{r} {state['node_attempts'].get(r, 0)}/{state['max_research_attempts']}" for r in ROLES)
                details = ('보고서 작성 3회 이후에도 승인 조건을 충족하지 못했습니다.' if policy['remaining_attempts']['report_generator'] == 0 else
                           f"필요한 조사 근거 미확보({', '.join(missing)}). 추가 수집 가능 작업이 없습니다.")
                details += ' 수집 횟수: ' + counts
                if review:
                    details += ' 검토 지적: ' + '; '.join(review.get('required_actions', []))
                return stop(state, 'NO_AVAILABLE_ACTION', details, step_count=step)
            else:
                payload = {'company_name': state['company_name'], 'request': state['research_request'],
                           'research': policy['research'], 'report': report, 'report_review': review,
                           'report_review_status': policy['report_review_status'],
                           'current_evidence_sufficient': state['evidence_sufficient'] if policy['report_current'] else False,
                           'overview_ready': policy['healthy'],
                           'node_status': state['node_status'], 'node_attempts': state['node_attempts'],
                           'node_errors': state['node_errors'], 'available_agents': policy['available_agents'],
                           'allowed_actions': policy['allowed_actions'], 'remaining_attempts': policy['remaining_attempts'],
                           'last_decision_error': state['last_error']}
                output = supervisor_handler(payload) if supervisor_handler else import_module('agents.supervisor').run(payload)
                decision = ResearchDecision.model_validate(output)
                validate_plan(decision, policy)
            log(state, 'decision', decision=decision.action, reason=decision.reason, plan=decision.model_dump())
            # 과거 초안/평가 지적은 재작성 입력으로 보존한다. 승인/검토 가능 여부는
            # report_basis와 review_revision을 통해 현재 근거 버전만 인정한다.
            report_update = {}
            return {**report_update, 'action': decision.action,
                    'status': 'finished' if decision.action == 'finish' else 'blocked' if decision.action == 'blocked' else 'running',
                    'evidence_sufficient': (policy['healthy'] if decision.action == 'write' else
                                            state['evidence_sufficient'] if decision.action in ('review', 'finish') else False),
                    'verdict': ({'action': decision.action, 'reason': decision.reason,
                                 'evidence_sufficient': decision.action == 'finish', 'missing_items': decision.missing_items}
                                if decision.action in ('finish', 'blocked') else None),
                    'step_count': step, 'decision_failures': 0,
                    'research_tasks': {t.agent: t.model_dump() for t in decision.tasks},
                    'last_error': 'EVIDENCE_INSUFFICIENT' if decision.action == 'blocked' else None}
        except Exception as exc:
            from agents.collection import safe_error_details
            validation_details = safe_error_details(exc)
            detail = str(exc) if isinstance(exc, PlanError) else json.dumps(validation_details, ensure_ascii=False) if validation_details else type(exc).__name__
            failures = state['decision_failures'] + 1
            log(state, 'decision_retry', reason=detail, error_type=type(exc).__name__, attempt=failures)
            if failures >= 3:
                return {**stop(state, 'DECISION_RETRY_LIMIT', f'허용 행동에 맞는 계획 생성 실패: {detail}', step_count=step), 'decision_failures': failures}
            return {'action': 'replan', 'status': 'running', 'verdict': None, 'research_tasks': {},
                    'last_error': detail, 'step_count': step, 'decision_failures': failures}

    def report_task(state):
        role = 'report_generator'
        attempt = state['node_attempts'].get(role, 0) + 1
        payload = {'company_name': state['company_name'], 'request': state['research_request'],
                   'research': state['perspectives'], 'report_date': state['report_date'],
                   'report': state['report'], 'report_review': state['report_review'],
                   'execution_errors': state['node_errors']}
        try:
            output = report_handler(payload) if report_handler else import_module('agents.report_generator').run(payload)
            registry = evidence_registry(state['perspectives'])
            draft = normalize_citations(output['draft'], registry)
            report = {'company_name': state['company_name'], 'draft': draft.model_dump(),
                      'markdown': render_markdown(draft, state['company_name'], state['report_date'], registry)}
            log(state, 'task_end', node=role, status='success')
            return {'report': report, 'report_revision': state['report_revision'] + 1,
                    'report_review': None, 'review_revision': None,
                    'report_basis': {r: state['node_attempts'].get(r, 0) for r in ROLES},
                    'node_status': {role: 'success'}, 'node_attempts': {role: attempt}, 'node_errors': {role: None}}
        except Exception as exc:
            from agents.collection import safe_error_details
            log(state, 'task_error', node=role, error_type=type(exc).__name__, validation_details=safe_error_details(exc))
            return {'report_review': None, 'review_revision': None, 'node_status': {role: 'error'},
                    'node_attempts': {role: attempt}, 'node_errors': {role: type(exc).__name__}}

    def review_task(state):
        role = 'reviewer'
        attempt = state['node_attempts'].get(role, 0) + 1
        payload = {'company_name': state['company_name'], 'research': state['perspectives'], 'report': state['report']}
        try:
            output = reviewer_handler(payload) if reviewer_handler else import_module('agents.reviewer').run(payload)
            review = ReportReview.model_validate(output).model_dump()
            log(state, 'task_end', node=role, status='success')
            return {'report_review': review, 'review_revision': state['report_revision'],
                    'node_status': {role: 'success'}, 'node_attempts': {role: attempt}, 'node_errors': {role: None}}
        except Exception as exc:
            from agents.collection import safe_error_details
            log(state, 'task_error', node=role, error_type=type(exc).__name__, validation_details=safe_error_details(exc))
            return {'node_status': {role: 'error'}, 'node_attempts': {role: attempt}, 'node_errors': {role: type(exc).__name__}}

    def route(state):
        if state['action'] == 'replan':
            return 'supervisor'
        if state['action'] == 'collect':
            return [Send(role, dict(state)) for role in state['research_tasks']]
        return {'write': 'report_generator', 'review': 'reviewer'}.get(state['action'], END)

    builder = StateGraph(State)
    builder.add_node('supervisor', supervisor)
    for role in ROLES:
        builder.add_node(role, workers.get(role) or import_module(f'agents.{role}').as_node())
        builder.add_edge(role, 'supervisor')
    builder.add_node('report_generator', report_task)
    builder.add_node('reviewer', review_task)
    builder.add_edge('report_generator', 'supervisor')
    builder.add_edge('reviewer', 'supervisor')
    builder.add_edge(START, 'supervisor')
    builder.add_conditional_edges('supervisor', route)
    # 조사·보고서 payload는 State에만 유지하고 외부 파일/체크포인트는 사용하지 않는다.
    return builder.compile()
