"""검증된 입력을 그대로 인용하는 구조화 보고서. 최종 LLM 문체 생성은 미연결."""

from .evaluation_support import (
    ReportAgentInput, ReportResult, collect_evidence, mapping,
)

# TODO(프롬프트 연동): 프롬프트 담당자의 최종 보고서 작성 프롬프트를 아래에 붙여넣는다.
# 현재 이 상수는 사용하지 않는다. LLM/API 확정 후 호출부를 별도로 연결해야 한다.
REPORT_SYSTEM_PROMPT = """
TODO: 프롬프트 담당자가 작성한 최종 프롬프트를 여기에 붙여넣는다.
"""


class ReportGenerator:
    def invoke(self, agent_input: ReportAgentInput | dict) -> ReportResult:
        # RAG 결과 수신 지점: Graph에서 종합 판단에 사용한 것과 같은 market_result와
        # tech_result를 전달받는다. 보고서 작성을 위해 새 RAG 검색을 실행하지 않는다.
        data = ReportAgentInput(**mapping(agent_input, "report input"))
        evidence, aliases = collect_evidence(data.results)
        used = set()

        # TODO(프롬프트/LLM 연동): 최종 프롬프트 수령 후 아래 템플릿 본문 생성 부분에
        # 검증된 입력만 사용하는 LLM 호출을 연결한다. 기존 점수·판단은 그대로 유지하고,
        # 생성 본문의 실제 인용 ID를 검증해 used와 references를 구성해야 한다.
        def section(result):
            lines = []
            for item in result["evaluations"]:
                score = "미제공" if item["score"] is None else str(item["score"])
                ids = list(dict.fromkeys(aliases[eid] for eid in item["evidence_ids"]))
                used.update(ids)
                citations = " ".join(f"[근거:{eid}]" for eid in ids)
                lines.append(f"{item['criterion']} ({score}점): {item['reason']} {citations}".strip())
            return "\n".join(lines) or "평가 결과 미제공"

        founder = section(data.founder_result)
        market = section(data.market_result)
        technology = section(data.tech_result)
        investment = data.investment_result
        score = "미산정" if investment.total_score is None else str(investment.total_score)
        verdict = f"종합 점수: {score} / 100. 판단: {investment.decision}."
        review = [verdict, *investment.key_reasons]
        review.extend(f"위험: {risk}" for risk in investment.risks)
        review.extend(f"추가 확인: {item}" for item in investment.missing_information)
        review.append(f"추가 정보 필요: {investment.needs_more_information}")
        return ReportResult(
            summary=f"평가 대상: {data.company_name}. {verdict}",
            # TODO(RAG 입력 합의): 핵심 사업·고객·수익 구조와 해당 근거 ID를 어떤 필드로
            # 받을지 RAG/공통 모델 담당자와 정한 뒤 사업 개요에 연결한다. 현재 필드는 미정이다.
            business_overview=(f"기업명: {data.company_name}. "
                               "사업 개요 전용 입력이 없어 핵심 사업·고객·수익 구조는 별도 확인이 필요합니다."),
            market_analysis=market,
            product_technology_and_team=f"제품·기술\n{technology}\n창업자·팀\n{founder}",
            investment_review="\n".join(review),
            references=[item for item in evidence if item["evidence_id"] in used],
        )
