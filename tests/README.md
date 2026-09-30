# 종합 투자 판단·보고서 Agent 연동 안내

저장소 루트(`skala-rag`)에서 실행:

```sh
python3 -m unittest discover -s tests -v
```

테스트 데이터는 실제 기업 평가가 아닌 연동 검증용 Mock이다. 외부 API, 패키지 설치, API 키가 필요하지 않다.

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

현재는 결정적 통합 로직과 템플릿 보고서이며 LLM 호출은 연결하지 않았다. `INVESTMENT_SYSTEM_PROMPT`, `REPORT_SYSTEM_PROMPT`에 최종 프롬프트를 붙여넣을 위치가 있지만 상수는 아직 실행에 사용되지 않는다. 모델/API 합의 후 호출을 연결해야 한다. Graph, State, Branch, Loop, RAG, 창업자 Agent는 구현 범위에 포함하지 않았다.

근거 ID와 연결 여부는 검증하지만, 주장·출처의 의미적 관련성, 사실성, 자료 간 모순을 검증하지는 않는다. `pending`은 근거 충분성 또는 투자 적합성을 인증하는 결과가 아니다. 이 검토 기준과 최종 판단 기준은 팀 합의가 필요하다. `confidence`는 형식만 검증하고 임의 임계값으로 판단에 사용하지 않는다.

사업 개요 전용 필드가 현재 공통 입력에 없어 해당 장에는 기업명과 추가 확인 안내를 표시한다. 최종 프롬프트/입력 계약 확정 후 핵심 사업·고객·수익 구조를 근거와 연결해야 한다. SUMMARY는 기업명·총점·판단만 표시한다. 페이지 배치나 5페이지 파일 출력은 제공하지 않는다.

연동 자료를 받으면 아래 파일과 위치를 수정한다. 경로는 저장소 루트 기준이다.

| 받은 자료 | 수정 파일 및 위치 |
|---|---|
| 최종 종합 투자 판단 프롬프트 | `agents/investment_evaluator.py`의 `INVESTMENT_SYSTEM_PROMPT`, `InvestmentEvaluator.invoke()` |
| `prompts/report_generator_prompt.py`의 확정된 보고서 프롬프트 | `agents/report_generator.py`의 `REPORT_SYSTEM_PROMPT`, `ReportGenerator.invoke()` |
| `agents/founder_insight.py`, `agents/market_scout.py`, `agents/tech_brief.py`의 실제 출력 및 공통 모델 | `agents/evaluation_support.py`의 `validate_specialist()`, `InvestmentAgentInput` (공통 모델 정의 경로는 미정) |
| RAG 결과의 사업 개요 필드 및 근거 ID | `agents/evaluation_support.py`의 입력 모델, `agents/report_generator.py`의 `business_overview` |
| Graph/State 연결 계약 | README에 예정된 `graph.py`, `state.py`에서 입력 전달·결과 저장·`InputValidationError` 처리 연결 |

현재 Mock 검증은 `tests/test_person2_agents.py`에 있다. 실제 출력 계약이 바뀌면 해당 테스트의 `mock_input()`과 연동 검증도 함께 수정한다.
