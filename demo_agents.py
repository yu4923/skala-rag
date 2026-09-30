"""투자 판단·보고서 Agent 실행 예제. 모든 입력은 실제 기업과 무관한 Mock이다."""

import argparse
import json

from agents.evaluation_support import CRITERIA, ReportAgentInput, ModelConfigurationError, resolve_model
from agents import investment_evaluator
from agents.investment_evaluator import InvestmentEvaluator
from agents.report_generator import ReportGenerator


def build_mock_input(needs_more_information=False):
    data = {"company_name": "연동 테스트 기업 (Mock)"}
    for field, role in (("founder_result", "founder"), ("market_result", "market"),
                        ("tech_result", "technology")):
        evaluations, evidence = [], []
        for index, (criterion, maximum) in enumerate(CRITERIA[role].items()):
            evidence_id = f"{role}-demo-{index}"
            evaluations.append({
                "criterion": criterion, "score": maximum,
                "reason": "실제 평가가 아닌 실행 확인용 Mock 평가",
                "evidence_ids": [evidence_id],
            })
            evidence.append({
                "evidence_id": evidence_id, "claim": f"{criterion} 연동 확인용 Mock 근거",
                "source_title": "실행 예제용 가상 문서", "source_url": None, "page": 1,
            })
        data[field] = {
            "agent_name": role, "company_name": data["company_name"],
            "evaluations": evaluations, "evidence": evidence, "risks": [],
            "missing_information": [], "confidence": "medium",
            "needs_more_information": False,
        }
    if needs_more_information:
        market = data["market_result"]
        market["evaluations"][1]["score"] = None
        market["evaluations"][1]["evidence_ids"] = []
        market["missing_information"] = ["실제 고객 수요 자료 미제공 (Mock 시나리오)"]
        market["needs_more_information"] = True
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--needs-more-information", action="store_true",
                        help="고객 수요 점수와 근거가 부족한 Mock 시나리오 실행")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--llm", action="store_true", help=".env 모델로 실제 LLM 두 번 호출 (API 비용 발생, 입력은 Mock)")
    mode.add_argument("--check-config", action="store_true", help=".env 모델 생성 설정만 확인 (API 호출 없음)")
    args = parser.parse_args()
    model = None
    if args.llm or args.check_config:
        try:
            model = resolve_model(None, investment_evaluator.MODEL,
                                  investment_evaluator.MODEL_FACTORY, investment_evaluator.MODEL_SETTINGS)
            if model is None:
                raise ModelConfigurationError("실제 호출용 모델 팩터리가 설정되지 않았습니다")
        except ModelConfigurationError as exc:
            parser.error(str(exc))
    if args.check_config:
        print("모델 생성 설정 확인 완료. API 키 유효성·모델 접근 권한은 아직 검증하지 않았습니다.")
        return
    data = build_mock_input(args.needs_more_information)
    investment = InvestmentEvaluator(model=model).invoke(data)
    report = ReportGenerator(model=model).invoke(ReportAgentInput(**data, investment_result=investment))
    print(json.dumps({
        "notice": ("Mock 입력을 실제 LLM으로 처리한 결과입니다. 실제 기업 평가나 RAG 결과가 아닙니다."
                   if args.llm else "Mock 실행 결과입니다. 실제 기업 평가 또는 LLM/RAG 실행 결과가 아닙니다."),
        "investment_result": investment.model_dump(),
        "report_result": report.model_dump(),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
