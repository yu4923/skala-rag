import argparse
from uuid import uuid4
from graph.graph import (
    ENABLE_LANGSMITH,
    GRAPH_RECURSION_LIMIT,
    build_graph,
    configure_logging,
    make_initial_state,
    setting,
)

COMPANIES = [name.strip() for name in setting("COMPANIES").split(",")]
DEFAULT_REQUEST = setting("DEFAULT_REQUEST")


def initialize_rag_stores():
    """실행 시 누락된 로컬 인덱스를 구축하고 완성된 인덱스는 재사용한다."""
    from rag.tech_ingest import COMPANIES as TECH_COMPANIES, INDEX, build_vector_db
    from rag.market_scout import initialize_market_rag

    files = [INDEX / "manifest.json"]
    for code in TECH_COMPANIES:
        files.extend((INDEX / code / "index.faiss", INDEX / code / "index.pkl"))
    if not all(path.is_file() for path in files):
        if INDEX.exists():
            archived = INDEX.with_name(f"{INDEX.name}.incomplete-{uuid4().hex[:8]}")
            INDEX.rename(archived)
            print(f"미완성 기술 RAG 인덱스를 {archived}에 보존했습니다.", flush=True)
        print("기술 RAG 저장소를 구축합니다.", flush=True)
        build_vector_db()
    initialize_market_rag()


def main():
    parser = argparse.ArgumentParser(description="에너지 스타트업 투자 평가")
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=ENABLE_LANGSMITH)
    args = parser.parse_args()

    configure_logging(args.trace)
    initialize_rag_stores()
    initial_state = make_initial_state(COMPANIES, DEFAULT_REQUEST)
    result = initial_state
    shown = set()
    for result in build_graph().stream(
        initial_state, config={"recursion_limit": GRAPH_RECURSION_LIMIT}, stream_mode="values"
    ):
        company = result["company_context"]["company_name"]
        for field, label in (("founder_result", "창업자"), ("market_result", "시장"), ("tech_result", "기술"),
                             ("investment_result", "종합 평가")):
            value = result.get(field, {})
            key = (company, value.get("round_no"), field)
            if value.get("company_name") == company and key not in shown:
                print(f"[{company} / {value['round_no']}회차] {label} 완료", flush=True)
                shown.add(key)
    if "report" in result:
        print(result["report"])
    else:
        for error in result["errors"]:
            print(f"[{error['company_name']} / {error['node']}] {error['code']}: {error['message']}")
        raise SystemExit("보고서 생성이 완료되지 않았습니다. errors 기록과 Agent 구현을 확인하세요.")


if __name__ == "__main__":
    main()
