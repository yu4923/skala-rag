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
        if state['step_count'] >= state['max_steps']:
            return stop(state, 'STEP_LIMIT', 'Supervisor 실행 상한 도달. 승인되지 않은 보고서는 최종 결과로 확정하지 않음.')
        step = state['step_count'] + 1
        report = state['report']
        review = state['report_review']
        review_revision = state['review_revision']
        # 추가 수집 시 기존 보고서/검토를 무효화한다. 고정 실행 횟수를 사용하지 않는다.
        if state['action'] == 'collect' and state['research_tasks']:
            report, review, review_revision = None, None, None
        report_update = {'report': report, 'report_review': review, 'review_revision': review_revision}
        try:
            research = {}
            for role, perspective in state['perspectives'].items():
                result = Collection.model_validate(perspective)
                if role not in ROLES or result.company_name != state['company_name'] or result.agent != role:
                    raise ValueError('수집 결과의 기업 또는 Agent 불일치')
                research[role] = result.model_dump()
            review_status = ('missing_report' if not report else
                             'unreviewed' if not review or review_revision != state['report_revision'] or state['node_status'].get('reviewer') == 'error' else
                             'passed' if ReportReview.model_validate(review).passed else 'failed')
            available = [role for role in (*ROLES, 'report_generator', 'reviewer')
                         if state['node_attempts'].get(role, 0) < state['max_retries'] + 1]
            payload = {'company_name': state['company_name'], 'request': state['research_request'],
                       'perspectives': research, 'research': research, 'report': report,
                       'report_review': review, 'report_review_status': review_status,
                       'node_status': state['node_status'], 'node_attempts': state['node_attempts'],
                       'node_errors': state['node_errors'], 'available_agents': available,
                       'remaining_steps': state['max_steps'] - state['step_count']}
            output = supervisor_handler(payload) if supervisor_handler else import_module('agents.supervisor').run(payload)
            decision = ResearchDecision.model_validate(output)
            actual = {r for r, data in research.items() if data['findings'] and data['evidence']}
            if set(decision.collected_perspectives) != actual:
                raise ValueError('수집된 관점 선언과 실제 근거 불일치')
            sufficient = decision.evidence_sufficient and actual == set(ROLES) and not any(state['node_status'].get(r) == 'error' for r in ROLES)
            # 충분성 완료 요청은 현재 산출물 상태에 맞춰 작성/검토로 보정한다.
            # 미검토 보고서를 보류하거나 승인으로 간주하지 않는다.
            if decision.action == 'finish' and sufficient and review_status in ('missing_report', 'unreviewed'):
                target = 'write' if review_status == 'missing_report' else 'review'
                log(state, 'decision_corrected', requested_action='finish', decision=target, reason=f'현재 보고서 상태: {review_status}')
                decision = decision.model_copy(update={'action': target})
            if decision.action == 'write' and not sufficient:
                raise ValueError('보고서 작성에는 현재 세 관점의 근거 충분성 판정이 필요함')
            if decision.action == 'review' and (not report or review_status != 'unreviewed'):
                raise ValueError('검토할 현재 미검토 보고서가 필요함')
            if decision.action == 'finish' and (not sufficient or review_status != 'passed' or state['node_status'].get('report_generator') == 'error'):
                raise ValueError('finish에는 현재 보고서의 세 기준 검토 통과가 필요함')
            targets = [t.agent for t in decision.tasks] if decision.action == 'collect' else (
                ['report_generator'] if decision.action == 'write' else ['reviewer'] if decision.action == 'review' else [])
            if any(role not in available for role in targets):
                return {**stop(state, 'ATTEMPT_LIMIT', '선택된 작업의 재시도 상한 도달. 승인 보고서 없음.', step_count=step), **report_update}
            log(state, 'decision', decision=decision.action, reason=decision.reason, plan=decision.model_dump())
            return {**report_update, 'action': decision.action,
                    'status': 'finished' if decision.action == 'finish' else 'blocked' if decision.action == 'blocked' else 'running',
                    'evidence_sufficient': sufficient if decision.action != 'collect' else False,
                    'verdict': ({'action': decision.action, 'reason': decision.reason, 'evidence_sufficient': sufficient,
                                 'missing_items': decision.missing_items} if decision.action in ('finish', 'blocked') else None),
                    'step_count': step, 'research_tasks': {t.agent: t.model_dump() for t in decision.tasks},
                    'last_error': 'EVIDENCE_INSUFFICIENT' if decision.action == 'blocked' else None}
        except Exception as exc:
            return {**stop(state, type(exc).__name__, f'Supervisor 실행 또는 계약 검증 실패: {type(exc).__name__}', step_count=step), **report_update}

    def report_task(state):
        role = 'report_generator'
        attempt = state['node_attempts'].get(role, 0) + 1
        payload = {'company_name': state['company_name'], 'request': state['research_request'],
                   'research': state['perspectives'], 'report_date': state['report_date'],
                   'report': state['report'], 'report_review': state['report_review']}
        try:
            output = report_handler(payload) if report_handler else import_module('agents.report_generator').run(payload)
            registry = evidence_registry(state['perspectives'])
            draft = normalize_citations(output['draft'], registry)
            report = {'company_name': state['company_name'], 'draft': draft.model_dump(),
                      'markdown': render_markdown(draft, state['company_name'], state['report_date'], registry)}
            log(state, 'task_end', node=role, status='success')
            return {'report': report, 'report_revision': state['report_revision'] + 1,
                    'report_review': None, 'review_revision': None,
                    'node_status': {role: 'success'}, 'node_attempts': {role: attempt}, 'node_errors': {role: None}}
        except Exception as exc:
            log(state, 'task_error', node=role, error_type=type(exc).__name__)
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
            log(state, 'task_error', node=role, error_type=type(exc).__name__)
            return {'node_status': {role: 'error'}, 'node_attempts': {role: attempt}, 'node_errors': {role: type(exc).__name__}}

    def route(state):
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
