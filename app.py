import argparse
from graph.graph import (
    ENABLE_LANGSMITH,
    GRAPH_RECURSION_LIMIT,
    build_graph,
    configure_logging,
    make_initial_state,
)

COMPANIES = ["엔라이튼", "해줌", "식스티헤르츠", "시너지", "인코어드테크놀로지스", "브이피피랩", "브이젠", "에너지엑스", "크로커스에너지", "러셀"]
DEFAULT_REQUEST = "각 기업의 투자 가능성을 종합적으로 평가해서 보고서를 작성해줘"

def main():
    parser = argparse.ArgumentParser(description="에너지 스타트업 투자 평가")
    parser.add_argument("--companies", nargs="+", default=COMPANIES)
    parser.add_argument("--request", default=DEFAULT_REQUEST)
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=ENABLE_LANGSMITH)
    args = parser.parse_args()

    configure_logging(args.trace)
    initial_state = make_initial_state(args.companies, args.request)
    result = build_graph().invoke(initial_state, config={"recursion_limit": GRAPH_RECURSION_LIMIT})
    if "report" in result:
        print(result["report"])
    else:
        for error in result["errors"]:
            print(f"[{error['company_name']} / {error['node']}] {error['code']}: {error['message']}")
        raise SystemExit("보고서 생성이 완료되지 않았습니다. errors 기록과 Agent 구현을 확인하세요.")


if __name__ == "__main__":
    main()
