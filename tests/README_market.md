# 시장성 검증 Agent

`agents/market_scout.py`는 `prompts/market_scout_prompt.py`의 `MARKET_SCOUT_PROMPT`를
그대로 사용합니다. JSON 중괄호를 포함한 프롬프트에 `.format()`을 적용하지 않습니다.
현재 프롬프트의 중복 대입(`MARKET_SCOUT_PROMPT = MARKET_SCOUT_PROMPT = ...`)은
Python에서 실행 가능하므로 원본을 수정하지 않았습니다.

## 설치와 호출

프로젝트 루트에서 `python -m pip install -r requirements-market.txt`로 설치합니다.
웹 검색 서비스와 PDF 인덱스는 담당자가 제공하는 함수를 주입합니다.
이 모듈은 PDF 임베딩·인덱스 생성이나 특정 검색 서비스 연결까지 구현하지 않습니다.
실행 모델은 도구 호출과 구조화 출력을 지원해야 합니다.

```python
import os
from langchain_openai import ChatOpenAI
from agents.market_scout import MarketAgentInput, MarketSource, create_market_scout

# web_search, rag_search는 아래 계약대로 구현한 함수입니다.
agent = create_market_scout(
    model=ChatOpenAI(model=os.environ["LLM_MODEL"], temperature=0, timeout=60, max_retries=2),
    web_search=web_search,
    rag_search=rag_search,
)
data = MarketAgentInput(
    company_name="평가할 기업명",
    evaluation_request="국내 시장의 성장성과 유료 고객·계약 근거를 검증해줘.",
    company_description="기업의 공개 사업 설명",
    target_market="국내 가정용 에너지 관리 시장",
)
result = agent.invoke(data)
print(result.model_dump_json(indent=2))
```

이 예제에는 `OPENAI_API_KEY`, `LLM_MODEL`이 필요합니다. `.env` 자동 로딩은 하지 않습니다.
다른 모델 공급자는 해당 LangChain 모델 객체를 주입하면 됩니다.

## 입력

| 필드 | 기본값 / 용도 |
|---|---|
| `company_name` | 필수 기업명 |
| `evaluation_request` | 필수 평가 요청 |
| `company_description` | 빈 문자열, 기업 설명 |
| `target_market` | `국내 B2C 에너지 시장` |
| `round_no` | 1, 재평가는 2까지 허용 |
| `previous_missing_items` | 이전 회차의 부족 정보, 초기 검색어에도 반영 |
| `input_evidence` | 선택적 기업 제출 자료, `source_type="input"` |
| `criteria` | 공통 `CRITERIA['market']`의 항목·배점. 다른 계약은 입력 오류 |

`invoke()`에는 같은 형태의 딕셔너리도 전달할 수 있습니다.
기업 설명 자체는 검증된 사실이 아니며, 근거로 사용할 기업 자료는 출처를 붙여 `input_evidence`에 넣습니다.

## 검색 담당자의 계약

검색 함수는 `query: str`을 받아 `Sequence[MarketSource]` 또는 같은 필드의 딕셔너리 목록을 반환합니다.
검색 결과가 없으면 `[]`, 연결 장애는 예외를 발생시키세요.

```python
# 아래 레코드는 형식 설명용 가상 자료입니다. 실제 검색 결과로 채우세요.
pdf_chunk = MarketSource(
    source_title="시장·정책 보고서",
    source_type="pdf",
    source="market-report.pdf",  # PDF 경로 또는 문서 ID
    page=7,                     # 원본 문서의 1-based 페이지
    excerpt="검색한 페이지의 실제 원문",
    published_date="2026-06-01", # 선택, 발행일
)
web_chunk = MarketSource(
    source_title="고객 계약 공시",
    source_type="web",
    source="https://example.org/contracts",
    excerpt="웹 검색에서 확보한 실제 원문",
)
```

RAG에는 시장·정책·규제·고객 수요 PDF만 연결합니다. PDF의 `page`는 필수이고,
웹은 HTTP(S) URL과 `page=None`이어야 합니다. 검색기가 0부터 페이지를 매기면 어댑터에서
1-based로 변환해야 합니다. `evidence_id`는 회사·출처·페이지·원문을 기준으로 Agent가 생성합니다.

최초 RAG 검색 1회와 웹 검색 1회 후 모델이 추가 검색을 선택합니다. 기본 검색 예산은
총 8회이고, 호출당 상위 3개 청크를 사용합니다. `max_results`는 1~5로 변경할 수 있습니다.
검색기의 정렬 품질, 문서 선정, 최신성 필터, 인덱스 구축은 검색 담당자의 책임입니다.
설계서대로 시장·정책 자료는 최근 3년 자료를 우선하고, 기업별 실제 수요와 연결해 검토하세요.

## 점수와 출력

**이 저장소의 종합 판단 계약은 이미 배점이 적용된 점수를 합산합니다.**

- `시장 규모·성장 가능성`: 0~25점
- `실제 고객 수요`: 0~25점
- 합계: 0~50점, 추가 가중치를 곱하지 않음
- 미평가 항목: `None`, 해당 회차의 합계도 `None`

프롬프트 응답은 `MarketAssessment`로 검증합니다. 기업명·회차·역할, 두 평가 항목,
최대 배점, 합계, 출처 ID, 원문 인용, 문서 위치·페이지를 대조합니다.
인용문은 공백 차이를 제외하고 검색 원문에 포함되어야 합니다.
검증된 점수와 실제 인용한 근거만 `AgentResult(agent_name="market", ...)`로 변환합니다.
PDF 식별자는 `source_title`에, 원본 페이지는 `page`에 보존하고,
`claim`에는 모델의 주장과 확인된 원문을 함께 넣습니다.

```python
# run과 invoke는 각각 한 번의 평가를 실행합니다. 같은 평가를 보려면 run 한 번만 호출하세요.
run = agent.run(data)
run.result             # 공통 AgentResult: 종합 판단 Agent로 전달
run.assessment         # 원본 구조화 응답: status, summary, scores, score 등
run.errors             # 회사·회차·노드·오류 유형, 비밀정보가 없는 메시지
run.retrieval_queries  # 이번 회차의 검색어 목록
run.calculations       # 계산 도구 입력·근거 ID·성장률·CAGR 기록
```

검색 일부가 실패하면 검증 가능한 점수는 보존하되 추가 정보를 요청합니다.
검색 근거가 전혀 없거나 모델·출처 검증에 실패하면 두 점수 모두 `None`으로 반환합니다.
LLM이 `error`를 반환한 경우에도 점수를 확정하지 않습니다.
`partial`/`missing_items`/미평가/도구 실패가 있으면 `needs_more_information=True`입니다.
입력 자료만으로 평가한 항목도 독립 검증을 요청합니다. 현재 신뢰도는 추가 확인이 필요하면
`low`, 그 외에는 `medium`입니다. 기계적 출처 검사만으로 `high`를 부여하지 않습니다.

원본 프롬프트에 별도 위험 목록 필드가 없어 공통 `risks`는 빈 목록이며,
설치 부담·영업주기·정책 등 시장 제약은 `summary`와 항목별 `reason`에 기록합니다.
`summary`는 공통 모델에 별도 필드가 없어 첫 평가 항목의 `reason`에도 보존합니다.

계산 도구는 양수 수치, 동일 단위·지역·시장 범위, 시작/종료 연도와 실제 근거 ID를 받습니다.
`CAGR = ((종료값 / 시작값) ** (1 / 기간) - 1) * 100`을 계산하며 감소율도 처리합니다.
수치가 원문의 무엇을 뜻하는지, 비교 범위가 같은지, 기업 주장과 독립 검증의 구분은 모델 검토에 의존합니다.
원문 일치나 계산 성공 자체가 사실성·의미적 관련성·시장 적합성을 보증하지는 않습니다.

## Graph와 재평가

```python
# Graph State에 market_input, market_result, errors를 선언합니다.
market_node = agent.as_node()
update = market_node({"market_input": data.model_dump()})
# {"market_result": AgentResult, "errors": [오류 딕셔너리, ...]}

# 추가 조사 여부와 다른 기업 선택은 전체 Graph가 담당합니다.
if run.result.needs_more_information and data.round_no == 1:
    second_input = MarketAgentInput(
        **{**data.model_dump(), "round_no": 2,
           "previous_missing_items": run.result.missing_information}
    )
```

동시 실행 노드가 있으면 Graph의 `errors`에는 리스트 누적 reducer를 사용하세요.
`as_node()`는 변경분만 반환합니다. 입력/설정 오류는 즉시 예외로 전달합니다.
Agent는 내부 검색 횟수와 실행 단계를 제한하고, 기업별 최대 1회 재평가는 Graph가 실행합니다.

## 검증

```bash
python -m unittest discover -s tests -v
python -m agents.market_scout
# agents 폴더 안에서는 python market_scout.py도 가능
```

모듈 직접 실행은 import 확인과 호출 안내만 출력합니다. 실 평가에는 모델·검색기 연결이 필요합니다.
테스트는 가상 자료와 모의 모델을 사용하며 API 키, 네트워크, PDF 다운로드가 필요하지 않습니다.
실 검색 품질·평가 정확도는 실제 PDF 및 웹 검색기를 연결한 뒤 별도로 측정해야 합니다.
