"""자료 수집 Supervisor 실행. State는 실행 메모리에만 유지한다."""
import argparse
from uuid import uuid4
from graph.graph import build_graph, make_initial_state, setting, configure_logging, ENABLE_LANGSMITH, GRAPH_RECURSION_LIMIT

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
    parser = argparse.ArgumentParser(description='기업 자료 수집 및 Supervisor 근거 충분성 판단')
    parser.add_argument('--company', help='단일 조사 기업. 생략하면 COMPANIES 후보 순서 사용')
    parser.add_argument('--request', default=None)
    parser.add_argument('--trace', action=argparse.BooleanOptionalAction, default=ENABLE_LANGSMITH)
    args = parser.parse_args()
    configure_logging(args.trace)
    initialize_rag_stores()
    companies = [args.company] if args.company else [s.strip() for s in setting('COMPANIES').split(',') if s.strip()]
    request = args.request or setting('RESEARCH_REQUEST', default='기업의 팀·시장·고객·제품·기술을 실제 출처와 함께 중립적으로 조사해줘')
    graph = build_graph()
    deferred_states = []
    for company in companies:
        initial = make_initial_state(company, request,
                                     max_steps=setting('MAX_WORKFLOW_STEPS', int, 100),
                                     max_retries=setting('MAX_RESEARCH_RETRIES', int, 2),
                                     max_research_attempts=setting('MAX_RESEARCH_ATTEMPTS', int, 6))
        print(f"[{company}] trace_id={initial['trace_id']}", flush=True)
        result = graph.invoke(initial, config={'recursion_limit': GRAPH_RECURSION_LIMIT,
                                             'metadata': {'trace_id': initial['trace_id']}})
        verdict = result['verdict']
        print(f"[{company}] {result['status']}: {verdict['reason']}", flush=True)
        if result['status'] == 'finished':
            print(result['report']['markdown'])
            from agents.pdf_export import save_report_pdf
            path = save_report_pdf(result['report']['markdown'], company, initial['report_date'])
            print(f'PDF: {path}', flush=True)
            return
        deferred_states.append(result)
    from agents.report_generator import render_deferred_report
    markdown = render_deferred_report(deferred_states)
    print(markdown)
    if deferred_states:
        from agents.pdf_export import save_report_pdf
        path = save_report_pdf(markdown, '기업별_조사_보류', deferred_states[0]['report_date'], deferred=True)
        print(f'PDF: {path}', flush=True)


if __name__ == '__main__':
    main()
