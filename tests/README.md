# 종합 투자 판단·보고서 Agent 연동 안내

## LangGraph 모듈 진입점

`agents/tech_brief.py`, `agents/investment_evaluator.py`, `agents/report_generator.py`에 모듈 레벨 `run(state, **kwargs)`를 제공한다. 기존 `graph/graph.py`는 `call_agent()`에서 이 함수를 찾고 결과를 State에 저장하므로, `run()` 자체는 State 갱신 딕셔너리로 한 번 더 감싸지 않는다.

| 진입점 | Graph에서 받는 값 | 반환값 |
|---|---|---|
| `tech_brief.run(state, criteria=...)` | 기업·평가 요청·재시도 회차, 선택 제품명/추가 질의 | Graph의 `AgentResult` 딕셔너리 (`tech_result`에 저장) |
| `investment_evaluator.run(state, total_score=..., criteria_met=...)` | 세 전문 결과와 근거 검토, Graph가 계산한 총점·기준 충족 여부 | 판단 설명 문자열 |
| `report_generator.run(state)` | 현재 기업 또는 `company_results`에 저장된 평가 기록 | Markdown 보고서 문자열 |

세 함수 모두 기존 클래스의 `invoke()`를 호출하며 점수 계산·검색·생성 로직을 복사하지 않는다. 공통 변환은 기존 `agents/evaluation_support.py`에 있다. 기업/회차가 다른 결과, 배점 불일치, Graph 합계 불일치는 오류로 반환하여 Graph의 기존 예외 처리에 맡긴다. 입력 State는 변경하지 않는다. 재시도 회차는 `retry_count + 1`로 반환한다. 제품·기술의 `None` 점수는 Graph가 허용하는 partial 결과에서 미완료 항목으로 생략하며 0점으로 바꾸지 않는다.

각 모듈 최상단의 설정 영역:

```python
MODEL = None           # 이미 만든 모델 인스턴스를 기본값으로 지정할 때 사용
MODEL_FACTORY = None   # 모델 생성 함수를 사용할 때 지정
MODEL_SETTINGS = {}    # 위 팩터리가 받는 모델명·온도 등 keyword arguments
# tech_brief.py에만 추가로 존재
RETRIEVER = None
```

외부 `model=` 주입 → 모듈 `MODEL` → `MODEL_FACTORY(**MODEL_SETTINGS)` 순으로 선택한다. 임의 제공자나 모델명은 추가하지 않았다. 실제 모델의 생성 함수와 모델명/옵션을 파일 상단에서 설정하면 `run()` 안의 코드를 수정할 필요가 없다. Graph 초기화 코드에서 이미 만든 객체를 각 모듈의 `MODEL`과 제품·기술 모듈의 `RETRIEVER`에 설정하는 방법도 가능하다. 모델·검색 객체는 직렬화되는 State에 넣지 않는다.

직접 어댑터를 확인할 때는 아래처럼 외부 주입할 수 있다.

```python
from agents import tech_brief, investment_evaluator, report_generator

tech_result = tech_brief.run(state, model=model, retriever=retriever)
reason = investment_evaluator.run(state, model=model, total_score=total, criteria_met=meets)
report = report_generator.run(state, model=model)
```

각 호출에는 해당 단계까지 완성된 State가 필요하다. 기존 Graph는 `run()`에 모델을 전달하지 않으므로 실제 Graph 실행 전에는 모듈 설정을 완료해야 한다. 모델이 없는 종합 판단/보고서는 기존 오프라인 동작을 유지하고, 제품·기술은 모델 또는 Retriever 미설정 시 명확한 설정 오류를 반환한다.

Graph의 `recommend`/`reject`는 공통 투자 결과에서 그대로 보존한다. `criteria_met`이 없는 기존 단독 실행의 pending/additional_research 동작은 유지한다. 보고서는 투자 판단이 실행되지 않은 보류 기록도 총점 미산정으로 표시하고, 여러 기업 기록을 각각 기존 보고서 Agent로 작성한다. 미실행 평가나 누락된 점수를 임의 생성하지 않는다.

검증은 `GraphAdapterTests`에 포함했다. 실제 `graph.graph` 디스패처/검증기 검사는 해당 폴더와 `langgraph` 패키지가 있을 때 실행하고, 없으면 그 통합 테스트만 건너뛴다. 현재 검증 환경에서는 설치 후 통합 테스트까지 실행했다.

남은 전체 Graph 연결 항목: `integrate_results()`가 별도로 호출하는 `investment_evaluator.review_evidence()`는 기존에 없으며 이번 run 어댑터 범위에서는 추가하지 않았다. 창업자/시장성 모듈의 Graph 진입점, 실제 모델·Retriever 설정도 각 담당 범위에서 연결해야 한다. 이 세 run의 통합 테스트 통과가 전체 파이프라인 완성을 의미하지는 않는다.

## 제품·기술 요약 Agent

제품·기술 구현은 기존 `agents/tech_brief.py`에 모았다. `evaluation_support.py`의 팀 공통 `AgentResult`, `Evidence`, `CriterionEvaluation`을 재사용한다. 다른 Agent나 Graph를 수정하지 않았다. 테스트도 새 파일 없이 기존 `tests/test_person2_agents.py`의 `ProductTechTests`에 추가했다.

```python
from agents.tech_brief import ProductTechAgentInput, create_product_tech_agent

# model: invoke(messages)를 지원하는 채팅 모델
# retriever: search(company_name=..., queries=...)를 지원하는 검색 객체
agent = create_product_tech_agent(model, retriever)
result = agent.invoke(ProductTechAgentInput(
    company_name="평가할 기업명",
    evaluation_request="제품 작동 원리와 공개 성능의 시험 조건 확인",
    product_names=[],
    additional_queries=[],
))
tech_result = result.model_dump()  # Graph에서 사용할 상태 필드는 팀에서 연결
```

`RetrievedDocument` 필드는 문서 ID, 청크 ID, 본문, 출처명, 선택 페이지·URL·검색 점수·metadata다. 기본 어댑터는 이 모델 또는 같은 필드의 딕셔너리 목록만 받는다. 문서 페이지는 1부터 시작하며 0부터 세는 실제 검색기는 어댑터에서 변환해야 한다.

실제 RAG를 받으면 `agents/tech_brief.py`의 `adapt_retrieval_result()`에 필드 매핑을 구현하거나, `create_product_tech_agent(model, retriever, adapter=변환함수)`로 주입한다. Retriever의 호출 방식이 다르면 `search(*, company_name, queries)`를 제공하는 래퍼에서 변환한다. 비동기 검색, Vector DB, 임베딩, 검색 순위 기준은 구현하지 않았다. 입력 `company_name`을 검색기에 넘기지만 실제 기업 필터 적용은 검색기가 담당한다. `metadata.company_name`이 있으면 다른 기업 문서는 제외한다.

선택 메타데이터 `source_type`은 `pdf`/`web`/`input`이다. 없으면 PDF 확장자·페이지·URL로 보수적으로 판별하므로 실제 어댑터에서 명시하는 것이 좋다. `metadata.provenance`에 `company` 또는 `independent`를 전달하면 프롬프트가 기업 주장과 외부 자료를 구분할 수 있다. 이 값과 원천 자료 식별 방식은 RAG 담당자와 확정해야 한다. 출처 성격 미확인 상태를 독립 검증으로 간주하지 않는다. 재인용 문서 개수·검색 점수로 confidence를 높이지 않는다.

근거 ID는 `technology:<문서 ID>:<청크 ID>:<페이지>`로 생성하며 ID 구성요소는 URL 인코딩한다. 공통 Evidence에 chunk_id 필드가 없어 ID에 보존한다. 반환 근거의 제목·URL·페이지는 검색 원장에서 가져오고 claim에는 검증된 원문 발췌를 보존한다. 모델이 만든 source/page가 검색값과 다르거나 excerpt가 원문에 없으면 실패한다. 제품 요약은 인용 근거가 있는 평가의 reason에 포함하고 위험은 공통 risks에 보존한다.

점수는 README 및 기존 `CRITERIA`의 두 항목 **제품·기술 정보의 구체성 15점 / 제품 차별성·성능 정보의 명확성 10점**을 사용한다. 다른 항목, 중복·누락 항목, 배점 변경, 범위 밖 점수, 잘못된 합계는 거부한다. 평가 불가 항목은 `None`을 유지한다. 점수별 세부 판정 기준이나 독립 검증의 실질적 충분성 기준은 새로 정하지 않았다.

오류/부족 정보 처리:

- 입력·설정 오류: Pydantic `ValidationError` 또는 `ValueError`.
- 검색 호출·어댑터·동일 청크 내용 충돌: `ProductTechRetrievalError` (원인은 `__cause__`).
- 검색 결과 없음, 빈 본문·출처명, PDF 페이지 누락: `missing_information` 및 `needs_more_information=True`. 사용 가능한 문서만 처리한다.
- 모델 호출/JSON 형식 오류: `AgentGenerationError`.
- 근거 ID·발췌·출처·페이지·평가 항목 검증 실패: `ProductTechEvidenceError` (`AgentGenerationError` 하위).

프롬프트는 앞서 정한 파일 import 방식을 유지해 `prompts/tech_brief_prompt.py`에서 읽는다. Agent의 `PRODUCT_TECH_SYSTEM_PROMPT`가 해당 상수의 별칭이다. 기존 프롬프트의 scores/evidence 반환 형식은 `TechAssessment`에서 검증한 뒤 공통 `AgentResult`로 변환한다. JSON 전송 지시로 근거 ID 보존, 근거 없는 점수와 회차의 null, 공통 risks 전달을 명시했다. 이는 최종 점수 루브릭이 아니다.

**추가 합의 사항**: RAG의 호출·반환 형식과 페이지 기준·출처 성격 메타데이터, Graph의 상태 필드와 오류 처리 및 round_no, 프롬프트의 null 점수/risks 반환 규칙과 시험 조건·재인용 처리, 종합 판단의 항목 식별자·15/10점 사용 방식. 현재 전문 평가의 자연어가 시장성 판단이나 원문에 없는 사실을 포함하는지 의미적으로 완전 검증하지는 못한다. 코드는 항목 범위와 인용 원문/메타데이터를 검증하며 모델의 추론 진실성·5페이지 같은 출력 품질을 보장하지 않는다.

```sh
python3 -m pip install -r requirements-agent.txt
python3 -m unittest discover -s tests -v
```

`ProductTechTests`는 Mock 검색/모델로 정상 결과, 검색·변환 오류, 빈 문서, 중복·충돌 청크, PDF 페이지, 웹 URL, 근거 ID/발췌 조작, 항목·점수·합계 오류, 자료 부족, 어댑터 교체, 공통 결과와 종합 Agent의 호환, 프롬프트 전달을 검사한다. 실제 RAG/LLM API 호출 테스트는 수행하지 않았다.

## LLM 호출 연결

두 Agent 모두 `model`을 주입하면 파일 프롬프트와 검증된 입력을 실제 `model.invoke(messages)`에 전달한다. `messages`는 `role`/`content` 딕셔너리 목록이다. 모델은 문자열 또는 문자열 `content`를 가진 응답을 반환해야 한다. 제공자·모델명·API 키 설정은 호출하는 애플리케이션에서 관리한다.

```python
# model: 애플리케이션에서 설정한 채팅 모델 인스턴스
investment = InvestmentEvaluator(model=model).invoke(investment_input)
report = ReportGenerator(model=model).invoke(ReportAgentInput(
    **investment_input.model_dump(), investment_result=investment,
))
```

선택 입력 `company_context`, `evidence_review`, `criteria_met`, `evaluation_settings`를 지원한다. 보고서의 `evaluation_status`는 생략하면 기존 decision을 사용하고, 전달하면 decision과 일치해야 한다. `criteria_met`은 설명용으로 전달하며 통과·보류 임계값이나 decision 매핑을 새로 정하지 않는다.

투자 Agent는 생성한 설명을 `key_reasons`에 추가한다. 보고서 Agent는 원본 프롬프트의 SUMMARY 및 1~4장을 JSON 필드에 대응시키는 전송 지시를 추가하고 실제 생성 본문을 반환한다. 총점·decision 메타데이터, JSON 필드, 본문 인용 ID와 사용 근거 목록의 일치를 검증한다. 실패는 `AgentGenerationError`로 전달하며 조용히 Mock 결과로 바꾸지 않는다. 자연어 본문에 포함된 모든 숫자·판단의 의미적 일치와 사실성을 자동 보증하지는 않는다. 출력 페이지 수 검증도 별도다.

모델을 생략한 기존 `demo_agents.py`는 오프라인 템플릿 실행이다. `tests/test_person2_agents.py`의 `LLMTests`는 Fake 모델로 실제 호출 경계와 응답 검증을 검사한다. 공통 LLM 호출 함수는 기존 `agents/evaluation_support.py`에 포함한다. 유료 API 호출은 수행하지 않았다.

아래의 과거 검토 기록 중 'LLM 미연결' 설명은 위 구현 이전 상태이며, 현재 실행 방법은 이 절을 따른다.

저장소 루트(`skala-rag`)에서 실행:

```sh
python3 demo_agents.py
python3 demo_agents.py --needs-more-information
python3 -m unittest discover -s tests -v
```

Python 3.10 이상이 필요하다. 실행 전 `python3 -m pip install -r requirements-agent.txt`로 공통 모델의 Pydantic 의존성을 설치한다. Windows에서는 `python3` 대신 `py -3`를 사용할 수 있다. 테스트 데이터는 실제 기업 평가가 아닌 연동 검증용 Mock이며 외부 API와 API 키가 필요하지 않다. `demo_agents.py`는 테스트 모듈에 의존하지 않는 독립 실행 예제이며, JSON 결과를 터미널에 출력한다.

## 공개 인터페이스

```python
from agents.investment_evaluator import InvestmentEvaluator, InvestmentAgentInput
from agents.report_generator import ReportGenerator, ReportAgentInput

investment_input = InvestmentAgentInput(
    company_name=company_name,
    founder_result=founder_result,
    market_result=market_result,
    tech_result=tech_result,
)
investment = InvestmentEvaluator().invoke(investment_input)
report = ReportGenerator().invoke(ReportAgentInput(
    **investment_input.model_dump(),
    investment_result=investment,
))
# Graph State에 저장할 직렬화 데이터
investment_data = investment.model_dump()
report_data = report.model_dump()
```

`invoke()`에는 같은 필드의 딕셔너리도 전달할 수 있다. 전문 결과는 전달 문서의 `AgentResult` 필드를 가진 딕셔너리 또는 `model_dump()`를 지원하는 모델로 받는다. 기존 의존성 설정이 없어 이번 입출력 모델은 Python 표준 dataclass로 구현했다. 공통 `AgentResult` 모델은 아직 정의되지 않았으며 정의 파일의 경로도 미정이다. `agents/evaluation_support.py`는 `agents/investment_evaluator.py`와 `agents/report_generator.py`에서 사용하는 입출력 모델, 검증 어댑터, 순수 함수를 담는다.

## 검증 및 점수 규칙

- 평가 항목명은 전달 문서에 지정된 5개 문자열을 정확히 사용한다. 담당 Agent별 항목/배점은 `CRITERIA`에 정의되어 있다.
- 필수 Agent 누락, 기업명/역할 불일치, 미합의 또는 중복 항목, 범위를 벗어난 점수, 근거 ID 오류는 `InputValidationError(ValueError)`로 전달한다. Graph가 이 예외를 잡아 오류 State에 기록할 수 있다.
- Agent는 있지만 평가 항목·점수·연결 근거가 부족하면 부족 정보를 반환한다. 미제공 점수는 0점으로 간주하지 않으며 필수 점수가 모두 있어야 총점을 계산한다.
- 상위 Agent의 부족 정보와 추가 정보 요청 플래그를 보존한다. 구조상 부족이 있으면 `additional_research`, 없으면 기준 확정 전까지 `pending`이다. 통과/보류 임계값은 구현하지 않았다.
- 같은 근거(주장, 자료 제목, URL, 페이지 동일)의 ID를 통합하고 보고서 인용을 대표 ID로 연결한다. 같은 자료의 다른 주장·페이지는 별도 근거로 보존한다. 동일 ID에 다른 근거가 있으면 오류다. 모든 Agent가 충돌 없는 ID를 전달해야 한다.
- 보고서는 전문 결과와 투자 총점의 일치를 검사한다. 입력의 점수·판단·평가 이유를 그대로 출력하고 실제 인용 근거만 references에 넣는다. 입력 객체는 변경하지 않는다.

## 현재 한계와 팀 연동 항목

현재는 결정적 통합 로직과 템플릿 보고서이며 LLM 호출은 연결하지 않았다. `REPORT_SYSTEM_PROMPT`는 `prompts/report_generator_prompt.py`의 `REPORT_GENERATOR_PROMPT`, `INVESTMENT_SYSTEM_PROMPT`는 `prompts/investment_evaluator_prompt.py`의 `INVESTMENT_EVALUATOR_PROMPT`를 import한 별칭이다. 프롬프트 본문은 해당 원본 파일에서만 관리한다. 변경 사항은 다음 프로세스 실행 시 반영되며 실행 중인 프로세스는 재시작해야 한다. 두 상수는 아직 LLM 호출에 사용되지 않는다. 모델/API 및 아래 출력 계약 합의 후 호출을 연결해야 한다. Graph, State, Branch, Loop, RAG, 창업자 Agent는 구현 범위에 포함하지 않았다.

근거 ID와 연결 여부는 검증하지만, 주장·출처의 의미적 관련성, 사실성, 자료 간 모순을 검증하지는 않는다. `pending`은 근거 충분성 또는 투자 적합성을 인증하는 결과가 아니다. 이 검토 기준과 최종 판단 기준은 팀 합의가 필요하다. `confidence`는 형식만 검증하고 임의 임계값으로 판단에 사용하지 않는다.

사업 개요 전용 필드가 현재 공통 입력에 없어 해당 장에는 기업명과 추가 확인 안내를 표시한다. 최종 프롬프트/입력 계약 확정 후 핵심 사업·고객·수익 구조를 근거와 연결해야 한다. SUMMARY는 기업명·총점·판단만 표시한다. 페이지 배치나 5페이지 파일 출력은 제공하지 않는다.

연동 자료를 받으면 아래 파일과 위치를 수정한다. 경로는 저장소 루트 기준이다.

| 받은 자료 | 수정 파일 및 위치 |
|---|---|
| 종합 투자 판단 프롬프트 본문 변경 | `prompts/investment_evaluator_prompt.py`의 `INVESTMENT_EVALUATOR_PROMPT`만 수정. LLM 호출은 `agents/investment_evaluator.py`의 `InvestmentEvaluator.invoke()`에 연결 |
| 보고서 프롬프트 본문 변경 | `prompts/report_generator_prompt.py`의 `REPORT_GENERATOR_PROMPT`만 수정. LLM 호출은 `agents/report_generator.py`의 `ReportGenerator.invoke()`에 연결 |
| `agents/founder_insight.py`, `agents/market_scout.py`, `agents/tech_brief.py`의 실제 출력 및 공통 모델 | `agents/evaluation_support.py`의 `validate_specialist()`, `InvestmentAgentInput` (공통 모델 정의 경로는 미정) |
| RAG 결과의 사업 개요 필드 및 근거 ID | `agents/evaluation_support.py`의 입력 모델, `agents/report_generator.py`의 `business_overview` |
| Graph/State 연결 계약 | README에 예정된 `graph.py`, `state.py`에서 입력 전달·결과 저장·`InputValidationError` 처리 연결 |

현재 Mock 검증은 `tests/test_person2_agents.py`에 있다. 실제 출력 계약이 바뀌면 해당 테스트의 `mock_input()`과 연동 검증도 함께 수정한다.

## 프롬프트 검토 및 요청 사항

아래 2~6번은 이전 프롬프트 4개를 기준으로 작성한 검토 기록이다. 최신 프롬프트를 가져왔으므로 다음 입출력 연동 작업에서 해결 여부를 다시 확인해야 한다. 1번은 새 종합 투자 판단 프롬프트를 기준으로 갱신했다. 다른 Agent의 프롬프트 원본을 임의로 수정하지 않았다.

1. **종합 투자 판단 프롬프트 import 완료, 실행 계약 연결 필요**: `prompts/investment_evaluator_prompt.py`를 Agent에서 직접 import한다. 이 프롬프트는 계산된 `total_score`, `criteria_met`, `evidence_review`, 판단 설정값 등을 받아 판단 근거 문자열을 반환한다. 현재 `InvestmentAgentInput`에는 `criteria_met`, `evidence_review`, 판단 설정값 필드가 없으므로 Graph와 전달 계약을 확정해야 한다. 문자열 결과는 향후 설명 필드에 연결하며 `InvestmentResult` 전체를 대체하지 않는다. LLM/API 연결 전까지 기존 결정적 로직과 `pending`/`additional_research` 동작을 유지한다.
2. **전문 Agent 출력 스키마 통일 요청**: `founder_insight_prompt.py`, `market_scout_prompt.py`, `tech_brief_prompt.py`는 단일 `score`, `conclusion`, `missing_items`, 근거의 `source`/`url`을 반환한다. 현재 입력은 `agent_name`, `company_name`, `evaluations`(항목·점수·이유·`evidence_ids`), `evidence`(`evidence_id`, `claim`, `source_title`, `source_url`, `page`), `risks`, `missing_information`, `confidence`, `needs_more_information`이 필요하다. 어느 계약을 사용할지 공통 모델과 함께 확정해야 한다. 단일 점수를 여러 항목으로 임의 분배할 수 없으므로 전문 프롬프트에 항목별 배점(25 / 25+25 / 15+10)과 미확인 점수 `null` 규칙을 요청한다.
3. **보고서 목차·출력 형식 확정 요청**: `report_generator_prompt.py`는 Markdown으로 '주요 위험과 한계'와 '종합 평가'를 분리한다. 기존 `ReportResult`는 `summary`, `business_overview`, `market_analysis`, `product_technology_and_team`, `investment_review`, `references`이며 위험은 `investment_review`에 포함한다. 기존 구조화 출력을 유지할지, 별도 위험 필드를 추가하고 Markdown으로 변환할지 확정이 필요하다. 최종 판단뿐 아니라 총점·항목 점수도 변경하지 않는 규칙을 명시해 달라고 요청한다.
4. **출처 연결 규칙 요청**: 본문 인용에 입력 `evidence_id`를 사용하고 존재하지 않는 ID를 만들지 않는 규칙, 실제 인용한 ID만 반환하는 규칙이 필요하다. 전체 입력 근거는 후보 자료이며 사용 출처는 본문 생성 후 결정된다. 현재 보고서 프롬프트의 '실제 사용된 출처 목록'이 전문 Agent의 사용 출처인지 최종 보고서의 사용 출처인지 구분해야 한다.
5. **사업 개요 입력 합의 요청**: 보고서에서 요구하는 핵심 사업·고객·수익 구조는 현재 공통 모델에 전용 필드가 없다. `tech_brief_prompt.py`의 `technology_summary`도 현재 보고서가 사용하지 않는다. RAG/공통 모델과 해당 내용 및 근거 ID를 전달할 위치를 정해야 한다.
6. **기술 평가 범위 확인 요청**: `tech_brief_prompt.py`는 '제품·기술 및 위험 검증'을 요구한다. README의 배점은 정보의 구체성·명확성 평가이며 기술 타당성이나 실제 성능의 독립 검증을 의미하지 않는다. 위험 정리와 실제 검증의 범위를 구분하고 점수가 의미하는 바를 맞춰야 한다.

LLM 모델/API, 근거 충분성·모순 판단 기준, 페이지 크기·글꼴 등 보고서 렌더링 규격도 별도 확정이 필요하다. '5페이지 이내/반 페이지 SUMMARY'는 프롬프트만으로 보장할 수 없으며 최종 출력 단계에서 확인해야 한다. 전문 프롬프트의 `{{ ... }}`는 템플릿 포맷팅용 이스케이프일 수 있으므로, 향후 호출부에서 `.format()` 사용 여부를 확인해야 한다.
