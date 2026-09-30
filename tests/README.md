# 종합 투자 판단·보고서 Agent 연동 안내

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
