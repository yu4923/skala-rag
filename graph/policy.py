"""분기 정책: 내용 충분성은 모델, 실행 가능성과 승인/상한은 코드의 책임이다."""
from agents.contracts import Collection, ReportReview, ROLES


class PlanError(ValueError):
    """외부 예외 원문과 구분되는 안전한 계획 검증 사유."""


def inspect_state(state):
    research = {}
    for role, value in state['perspectives'].items():
        result = Collection.model_validate(value)
        if role not in ROLES or result.company_name != state['company_name'] or result.agent != role:
            raise PlanError('조사 결과의 기업/관점이 현재 실행과 다릅니다.')
        research[role] = result.model_dump()
    covered = [r for r in ROLES if r in research and research[r]['findings'] and research[r]['evidence']]
    healthy = len(covered) == len(ROLES)  # 최근 추가 수집 실패가 기존 검증 자료를 무효화하지는 않는다
    limits = {**{r: state['max_research_attempts'] for r in ROLES},
              'report_generator': min(state['max_report_attempts'], 3),
              'reviewer': 3 * (state['max_retries'] + 1)}
    remaining = {r: max(0, n - state['node_attempts'].get(r, 0)) for r, n in limits.items()}
    available = [r for r in ROLES if remaining[r] > 0]
    basis = {r: state['node_attempts'].get(r, 0) for r in ROLES}
    report_current = bool(state['report']) and state.get('report_basis') == basis
    if not report_current:
        review_status = 'missing_report'
    elif not state['report_review'] or state['review_revision'] != state['report_revision'] or state['node_status'].get('reviewer') == 'error':
        review_status = 'unreviewed'
    else:
        review_status = 'passed' if ReportReview.model_validate(state['report_review']).passed else 'failed'
    approved = (healthy and state['evidence_sufficient'] and review_status == 'passed'
                and state['node_status'].get('report_generator') != 'error')
    if approved:
        allowed = ['finish']
    elif remaining['report_generator'] == 0 and review_status != 'unreviewed':
        allowed = ['blocked']
    elif healthy:
        # 개요 초안 작성과 최종 승인 충분성을 분리한다. 세부 정보 미확보는 초안 거부 사유가 아니다.
        allowed = []
        if review_status == 'unreviewed' and remaining['reviewer']:
            allowed.append('review')
        elif remaining['report_generator']:
            allowed.append('write')
        if available and remaining['report_generator'] > 0:
            allowed.append('collect')
        if not allowed:
            allowed = ['blocked']
    else:
        missing_roles = set(ROLES) - set(covered)
        allowed = ['collect'] if missing_roles.intersection(available) else ['blocked']
    return {'research': research, 'covered': covered, 'healthy': healthy, 'remaining_attempts': remaining,
            'available_agents': available, 'allowed_actions': allowed, 'report_current': report_current,
            'report_review_status': review_status, 'approved': approved, 'basis': basis}


def validate_plan(plan, policy):
    if plan.action not in policy['allowed_actions']:
        raise PlanError(f"현재 가능한 행동: {', '.join(policy['allowed_actions'])}")
    if plan.action == 'collect':
        invalid = [t.agent for t in plan.tasks if t.agent not in policy['available_agents']]
        if invalid:
            raise PlanError('호출 가능한 수집 Agent만 배정해야 합니다: ' + ', '.join(policy['available_agents']))
        if policy['healthy'] and policy['report_review_status'] != 'failed':
            ids = {e['evidence_id'] for d in policy['research'].values() for e in d['evidence']}
            if plan.research_need not in ('source_conflict', 'identity_unclear') or not plan.research_evidence_ids or set(plan.research_evidence_ids) - ids:
                raise PlanError('세 관점의 개요가 모였습니다. 추가 수집은 실제 근거 ID로 특정한 출처 충돌/기업 식별 문제를 해결할 때만 가능합니다. 그 외 한계는 보고서에 기술하세요.')
    if plan.action == 'write' and not policy['healthy']:
        raise PlanError('세 관점의 관련 조사 내용이 있어야 개요 초안을 작성할 수 있습니다.')
