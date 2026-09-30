import argparse
from graph.graph import (
    ENABLE_LANGSMITH,
    GRAPH_RECURSION_LIMIT,
    build_graph,
    configure_logging,
    make_initial_state,
)


def main():
    parser = argparse.ArgumentParser(description="에너지 스타트업 투자 평가")
    parser.add_argument("--companies", nargs="+", required=True, help="평가할 기업 목록")
    parser.add_argument("--request", required=True, help="투자 평가 요청")
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
