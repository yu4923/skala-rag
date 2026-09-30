"""모든 기업·점수·근거는 코드 검증 전용 Mock이며 실제 평가 자료가 아니다."""

from copy import deepcopy
import json
import unittest

from agents.evaluation_support import (
    CRITERIA, InputValidationError, InvestmentAgentInput, InvestmentResult,
    ReportAgentInput,
    AgentGenerationError,
)
from agents.investment_evaluator import InvestmentEvaluator, INVESTMENT_SYSTEM_PROMPT
from agents.report_generator import ReportGenerator, REPORT_SYSTEM_PROMPT


def mock_input():
    data = {"company_name": "연동 테스트 기업"}
    for field, role in (("founder_result", "founder"), ("market_result", "market"),
                        ("tech_result", "technology")):
        evaluations, evidence = [], []
        for index, (criterion, maximum) in enumerate(CRITERIA[role].items()):
            eid = f"{role}-test-{index}"
            evaluations.append(dict(criterion=criterion, score=maximum,
                                    reason="연동 테스트용 평가", evidence_ids=[eid]))
            evidence.append(dict(evidence_id=eid, claim=f"{role}-{index} 테스트 주장",
                                 source_title="테스트 문서", source_url=None, page=1))
        data[field] = dict(agent_name=role, company_name=data["company_name"],
                           evaluations=evaluations, evidence=evidence, risks=[],
                           missing_information=[], confidence="medium", needs_more_information=False)
    return data


class Person2Tests(unittest.TestCase):
    def setUp(self):
        self.data = mock_input()
        self.agent = InvestmentEvaluator()

    def report(self, data=None, investment=None):
        data = self.data if data is None else data
        investment = self.agent.invoke(data) if investment is None else investment
        return ReportGenerator().invoke(ReportAgentInput(**data, investment_result=investment))

    def test_complete_results_do_not_invent_threshold(self):
        result = self.agent.invoke(InvestmentAgentInput(**self.data))
        self.assertEqual(result.total_score, 100)
        self.assertEqual(result.decision, "pending")
        self.assertFalse(result.needs_more_information)
        self.assertEqual(result.missing_information, [])

    def test_missing_agents(self):
        for key in ("founder_result", "market_result", "tech_result"):
            with self.subTest(key=key):
                data = deepcopy(self.data)
                del data[key]
                with self.assertRaises(InputValidationError):
                    self.agent.invoke(data)
                with self.assertRaises(InputValidationError):
                    ReportAgentInput(**data, investment_result=InvestmentResult("연동 테스트 기업"))

    def test_missing_information_and_risks_are_merged(self):
        for key in ("founder_result", "market_result"):
            self.data[key]["risks"] = ["테스트 위험"]
            self.data[key]["missing_information"] = ["테스트 추가 확인"]
        result = self.agent.invoke(self.data)
        self.assertEqual(result.risks, ["테스트 위험"])
        self.assertEqual(result.missing_information, ["테스트 추가 확인"])
        self.assertTrue(result.needs_more_information)
        self.assertEqual(result.decision, "additional_research")

    def test_upstream_flag_without_details(self):
        self.data["market_result"]["needs_more_information"] = True
        result = self.agent.invoke(self.data)
        self.assertTrue(result.needs_more_information)
        self.assertTrue(result.missing_information)

    def test_missing_score_or_criterion_is_not_zero(self):
        for mode in ("score", "criterion"):
            with self.subTest(mode=mode):
                data = deepcopy(self.data)
                if mode == "score":
                    data["tech_result"]["evaluations"][0]["score"] = None
                else:
                    data["tech_result"]["evaluations"].pop()
                result = self.agent.invoke(data)
                self.assertIsNone(result.total_score)
                self.assertEqual(result.decision, "additional_research")
                self.assertIn("미산정", self.report(data).summary)

    def test_invalid_scores(self):
        for score in (-1, 26, float("nan"), float("inf"), True, "25"):
            with self.subTest(score=score):
                self.data["founder_result"]["evaluations"][0]["score"] = score
                with self.assertRaises(InputValidationError):
                    self.agent.invoke(self.data)

    def test_zero_and_decimal_scores(self):
        self.data["founder_result"]["evaluations"][0]["score"] = 0
        self.data["market_result"]["evaluations"][0]["score"] = 0.1
        self.data["market_result"]["evaluations"][1]["score"] = 0.2
        self.assertEqual(self.agent.invoke(self.data).total_score, 25.3)

    def test_invalid_evidence_reference(self):
        self.data["founder_result"]["evaluations"][0]["evidence_ids"] = ["missing"]
        with self.assertRaises(InputValidationError):
            self.agent.invoke(self.data)

    def test_no_evidence_is_additional_research(self):
        self.data["founder_result"]["evaluations"][0]["evidence_ids"] = []
        self.data["founder_result"]["evidence"] = []
        self.assertEqual(self.agent.invoke(self.data).decision, "additional_research")

    def test_duplicate_evidence_rewrites_citations(self):
        original = self.data["founder_result"]["evidence"][0]
        market = self.data["market_result"]
        market["evidence"][0] = dict(original, evidence_id="alias")
        market["evaluations"][0]["evidence_ids"] = ["alias"]
        report = self.report()
        self.assertEqual(len(report.references), 4)
        self.assertIn("[근거:founder-test-0]", report.market_analysis)
        self.assertNotIn("[근거:alias]", report.market_analysis)

    def test_same_source_different_claims_or_pages_are_preserved(self):
        self.assertEqual(len(self.report().references), 5)
        self.data["market_result"]["evidence"][1]["claim"] = self.data["market_result"]["evidence"][0]["claim"]
        self.data["market_result"]["evidence"][1]["page"] = 2
        self.assertEqual(len(self.report().references), 5)

    def test_unused_evidence_excluded(self):
        extra = dict(self.data["founder_result"]["evidence"][0], evidence_id="unused", claim="미인용 테스트")
        self.data["founder_result"]["evidence"].append(extra)
        self.assertNotIn("unused", [item["evidence_id"] for item in self.report().references])

    def test_id_collision_between_agents(self):
        self.data["market_result"]["evidence"][0]["evidence_id"] = "founder-test-0"
        self.data["market_result"]["evaluations"][0]["evidence_ids"] = ["founder-test-0"]
        with self.assertRaises(InputValidationError):
            self.agent.invoke(self.data)

    def test_wrong_company_role_duplicate_or_unknown_criterion(self):
        for mode in ("company", "role", "duplicate", "unknown"):
            with self.subTest(mode=mode):
                data = deepcopy(self.data)
                founder = data["founder_result"]
                if mode == "company":
                    founder["company_name"] = "다른 테스트 기업"
                elif mode == "role":
                    founder["agent_name"] = "market"
                elif mode == "duplicate":
                    founder["evaluations"].append(deepcopy(founder["evaluations"][0]))
                else:
                    founder["evaluations"][0]["criterion"] = "미합의 항목"
                with self.assertRaises(InputValidationError):
                    self.agent.invoke(data)

    def test_report_preserves_score_decision_and_inputs(self):
        original = deepcopy(self.data)
        result = self.agent.invoke(self.data)
        before = result.model_dump()
        report = self.report(investment=result)
        self.assertIn("100.0 / 100", report.investment_review)
        self.assertIn("pending", report.summary)
        self.assertEqual(result.model_dump(), before)
        self.assertEqual(self.data, original)
        self.assertEqual(len(report.model_dump()), 6)

    def test_report_rejects_changed_total(self):
        result = self.agent.invoke(self.data)
        result.total_score = 99
        with self.assertRaises(InputValidationError):
            self.report(investment=result)

    def test_serialized_graph_inputs(self):
        investment = self.agent.invoke(self.data).model_dump()
        report = ReportGenerator().invoke(dict(self.data, investment_result=investment))
        self.assertIn("pending", report.summary)

    def test_modified_model_revalidated(self):
        data = InvestmentAgentInput(**self.data)
        data.founder_result["evaluations"][0]["score"] = -1
        with self.assertRaises(InputValidationError):
            self.agent.invoke(data)


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.response


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.data = mock_input()
        self.investment = InvestmentEvaluator().invoke(self.data)
        self.output = dict(summary="테스트 요약", business_overview="테스트 사업",
                           market_analysis="테스트 시장 [근거:market-test-0]",
                           product_technology_and_team="테스트 기술", investment_review="판단 pending, 100점",
                           total_score=100.0, decision="pending", used_evidence_ids=["market-test-0"])

    def test_investment_calls_model_with_prompt_and_context(self):
        model = FakeModel("입력 근거에 대한 테스트 설명")
        self.data.update(criteria_met=None, evidence_review={"missing": ["검토 필요"]})
        result = InvestmentEvaluator(model=model).invoke(self.data)
        self.assertTrue(model.messages[0]["content"].startswith(INVESTMENT_SYSTEM_PROMPT))
        payload = json.loads(model.messages[1]["content"])
        self.assertEqual(payload["evidence_review"], self.data["evidence_review"])
        self.assertEqual(payload["total_score"], 100)
        self.assertIn(model.response, result.key_reasons)
        self.assertEqual(result.decision, "pending")

    def test_report_uses_generated_text_and_input_references(self):
        model = FakeModel(json.dumps(self.output))
        report = ReportGenerator(model=model).invoke(dict(self.data, investment_result=self.investment))
        self.assertEqual(report.summary, "테스트 요약")
        self.assertEqual(len(report.references), 1)
        self.assertTrue(model.messages[0]["content"].startswith(REPORT_SYSTEM_PROMPT))
        self.assertEqual(json.loads(model.messages[1]["content"])["evaluation_status"], "pending")

    def test_invalid_model_outputs_fail(self):
        for patch in ({"total_score": 50}, {"decision": "hold"},
                      {"used_evidence_ids": []},
                      {"used_evidence_ids": ["unknown"], "market_analysis": "[근거:unknown]"},
                      {"summary": ""}):
            with self.subTest(patch=patch):
                model = FakeModel(json.dumps(dict(self.output, **patch)))
                with self.assertRaises(AgentGenerationError):
                    ReportGenerator(model=model).invoke(dict(self.data, investment_result=self.investment))

    def test_invalid_json_and_model_failure(self):
        with self.assertRaises(AgentGenerationError):
            ReportGenerator(model=FakeModel("not JSON")).invoke(dict(self.data, investment_result=self.investment))
        class BrokenModel:
            def invoke(self, messages):
                raise RuntimeError("test failure")
        with self.assertRaises(AgentGenerationError):
            InvestmentEvaluator(model=BrokenModel()).invoke(self.data)


if __name__ == "__main__":
    unittest.main()
