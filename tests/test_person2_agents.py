"""모든 기업·점수·근거는 코드 검증 전용 Mock이며 실제 평가 자료가 아니다."""

from copy import deepcopy
import json
import os
import sys
from io import StringIO
import unittest

from agents.evaluation_support import (
    CRITERIA, InputValidationError, InvestmentAgentInput, InvestmentResult,
    ReportAgentInput,
    AgentGenerationError,
    create_model_from_env, ModelConfigurationError,
)
from agents.investment_evaluator import InvestmentEvaluator, INVESTMENT_SYSTEM_PROMPT
from agents.report_generator import ReportGenerator, REPORT_SYSTEM_PROMPT
from agents.tech_brief import (
    ProductTechAgent, ProductTechAgentInput, RetrievedDocument,
    ProductTechRetrievalError, ProductTechEvidenceError, evidence_id,
    create_product_tech_agent,
)
from agents.evaluation_support import AgentResult
from agents.evaluation_support import specialist_to_graph
from unittest.mock import patch


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


class MockProductTechRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.last_call = None

    def search(self, *, company_name, queries):
        self.last_call = {"company_name": company_name, "queries": queries}
        return self.documents


class MockProductTechModel:
    """응답은 전달된 테스트 문서만 사용한다. 실제 기업/성능 데이터를 만들지 않는다."""
    def __init__(self, mutate=None):
        self.mutate = mutate
        self.payload = None
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        self.payload = json.loads(messages[1]["content"])
        doc = self.payload["documents"][0]
        eid = doc["evidence_id"]
        output = {
            "company_name": self.payload["company_name"], "round_no": None,
            "agent": "tech_brief", "status": "success", "summary": "연동 테스트용 제품 요약",
            "evidence": [{"evidence_id": eid, "claim": doc["content"],
                          "excerpt": doc["content"], "source_type": doc["source_type"],
                          "source": doc["source"], "page": doc["page"]}],
            "scores": [{"criterion": item["criterion"], "score": item["max_score"],
                        "max_score": item["max_score"], "reason": "연동 테스트용 평가",
                        "evidence_ids": [eid]} for item in self.payload["criteria"]],
            "score": 25, "missing_items": [], "risks": ["연동 테스트용 추가 검토 위험"],
        }
        if self.mutate:
            self.mutate(output)
        return json.dumps(output, ensure_ascii=False)


class ProductTechTests(unittest.TestCase):
    def setUp(self):
        self.doc = RetrievedDocument(
            document_id="doc-test-001", chunk_id="chunk-test-001",
            content="테스트 제품의 구성과 작동 원리를 확인하기 위한 연동 테스트 문장",
            source_title="연동 테스트용 문서.pdf", page=1, retrieval_score=0.8,
            metadata={"source_type": "pdf", "provenance": "independent"},
        )
        self.request = ProductTechAgentInput(
            company_name="연동 테스트 기업", evaluation_request="제품 작동 원리 확인",
            product_names=["테스트 제품"], additional_queries=["추가 실증 조건"],
        )

    def run_agent(self, docs=None, mutate=None):
        self.retriever = MockProductTechRetriever([self.doc] if docs is None else docs)
        self.model = MockProductTechModel(mutate)
        return ProductTechAgent(self.model, self.retriever).invoke(self.request)

    def test_normal_result_and_downstream_compatibility(self):
        result = self.run_agent()
        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.agent_name, "technology")
        self.assertEqual([item.score for item in result.evaluations], [15, 10])
        self.assertFalse(result.needs_more_information)
        self.assertEqual(result.risks, ["연동 테스트용 추가 검토 위험"])
        data = mock_input()
        data["tech_result"] = result
        self.assertEqual(InvestmentEvaluator().invoke(data).total_score, 100)

    def test_prompt_queries_and_request_reach_model(self):
        self.run_agent()
        self.assertEqual(self.model.payload["evaluation_request"], self.request.evaluation_request)
        self.assertEqual(self.retriever.last_call["company_name"], self.request.company_name)
        self.assertTrue(any("추가 실증 조건" in query
                            for query in self.retriever.last_call["queries"]))
        from prompts.tech_brief_prompt import TECH_BRIEF_PROMPT
        self.assertTrue(self.model.messages[0]["content"].startswith(TECH_BRIEF_PROMPT))

    def test_empty_search_does_not_call_model(self):
        result = self.run_agent([])
        self.assertTrue(result.needs_more_information)
        self.assertTrue(all(item.score is None for item in result.evaluations))
        self.assertIsNone(self.model.payload)

    def test_retriever_error_is_not_empty_search(self):
        class BrokenRetriever:
            def search(self, **kwargs):
                raise RuntimeError("Mock 검색 오류")
        with self.assertRaises(ProductTechRetrievalError):
            ProductTechAgent(MockProductTechModel(), BrokenRetriever()).invoke(self.request)

    def test_blank_content_and_source_are_removed(self):
        for field in ("content", "source_title"):
            with self.subTest(field=field):
                doc = self.doc.model_dump()
                doc[field] = "  "
                result = self.run_agent([doc])
                self.assertTrue(result.needs_more_information)
                self.assertEqual(result.evidence, [])

    def test_duplicate_chunk_and_id_are_deduplicated(self):
        result = self.run_agent([self.doc, self.doc.model_copy(update={"retrieval_score": 0.5})])
        self.assertEqual(len(self.model.payload["documents"]), 1)
        self.assertEqual(len(result.evidence), 1)

    def test_conflicting_chunk_is_rejected(self):
        with self.assertRaises(ProductTechRetrievalError):
            self.run_agent([self.doc, self.doc.model_copy(update={"content": "다른 테스트 내용"})])

    def test_pdf_page_and_chunk_identity_preserved(self):
        result = self.run_agent()
        self.assertEqual(result.evidence[0].page, 1)
        self.assertEqual(result.evidence[0].evidence_id, evidence_id(self.doc))
        self.assertIn("chunk-test-001", result.evidence[0].evidence_id)
        self.assertEqual(result.evidence[0].claim, self.doc.content)

    def test_pdf_without_page_is_partial(self):
        result = self.run_agent([self.doc.model_copy(update={"page": None})])
        self.assertTrue(result.needs_more_information)
        self.assertTrue(any("PDF 페이지" in item for item in result.missing_information))
        self.assertIsNone(result.evidence[0].page)

    def test_invalid_citations_are_rejected(self):
        for field, value in (("evidence_id", "unknown"), ("source", "invented"),
                             ("page", 99), ("excerpt", "원문에 없는 테스트 주장")):
            with self.subTest(field=field):
                with self.assertRaises(ProductTechEvidenceError):
                    self.run_agent(mutate=lambda out: out["evidence"][0].update({field: value}))

    def test_missing_eval_reference_and_duplicate_evidence(self):
        for mutate in (lambda out: out["scores"][0].update(evidence_ids=["unknown"]),
                       lambda out: out["evidence"].append(deepcopy(out["evidence"][0]))):
            with self.assertRaises(ProductTechEvidenceError):
                self.run_agent(mutate=mutate)

    def test_market_criterion_and_missing_criterion_rejected(self):
        for mutate in (lambda out: out["scores"][0].update(criterion="시장 규모·성장 가능성"),
                       lambda out: out["scores"].pop()):
            with self.assertRaises(ProductTechEvidenceError):
                self.run_agent(mutate=mutate)

    def test_invalid_scores_and_totals(self):
        for score in (-1, 16, True, float("nan")):
            with self.subTest(score=score), self.assertRaises(ProductTechEvidenceError):
                self.run_agent(mutate=lambda out: out["scores"][0].update(score=score))
        with self.assertRaises(ProductTechEvidenceError):
            self.run_agent(mutate=lambda out: out.update(score=99))

    def test_missing_score_stays_none(self):
        def partial(out):
            out.update(status="partial", score=None, missing_items=["테스트 측정 조건 부족"])
            out["scores"][1].update(score=None, evidence_ids=[])
        result = self.run_agent(mutate=partial)
        self.assertIsNone(result.evaluations[1].score)
        self.assertTrue(result.needs_more_information)
        self.assertIn("테스트 측정 조건 부족", result.missing_information)

    def test_company_material_not_independent(self):
        doc = self.doc.model_copy(update={"metadata": {"source_type": "pdf", "provenance": "company"}})
        result = self.run_agent([doc])
        self.assertTrue(result.needs_more_information)
        self.assertEqual(result.confidence, "low")

    def test_adapter_can_be_replaced(self):
        raw = [{"text": self.doc.content}]
        def adapter(items):
            return [self.doc.model_copy(update={"content": item["text"]}) for item in items]
        agent = create_product_tech_agent(MockProductTechModel(), MockProductTechRetriever(raw), adapter=adapter)
        self.assertEqual(agent.invoke(self.request.model_dump()).agent_name, "technology")

    def test_bad_adapter_and_negative_page(self):
        for docs in ({"unexpected": []}, [dict(self.doc.model_dump(), page=0)]):
            with self.assertRaises(ProductTechRetrievalError):
                self.run_agent(docs)

    def test_invalid_json_or_empty_prompt(self):
        with self.assertRaises(AgentGenerationError):
            ProductTechAgent(FakeModel("not JSON"), MockProductTechRetriever([self.doc])).invoke(self.request)
        with patch("agents.tech_brief.PRODUCT_TECH_SYSTEM_PROMPT", ""):
            with self.assertRaises(ValueError):
                self.run_agent()

    def test_unsupported_summary_without_citations_is_discarded(self):
        def no_citations(out):
            out.update(evidence=[], score=None, summary="근거 없이 생성된 테스트 요약")
            for item in out["scores"]:
                item.update(score=None, evidence_ids=[])
        result = self.run_agent(mutate=no_citations)
        self.assertTrue(result.needs_more_information)
        self.assertNotIn("근거 없이 생성된 테스트 요약", str(result.model_dump()))

    def test_web_url_and_claim_are_preserved_from_registry(self):
        doc = self.doc.model_copy(update={
            "source_title": "연동 테스트 웹 문서", "page": None,
            "source_url": "https://example.com/test", "metadata": {"source_type": "web"},
        })
        result = self.run_agent([doc])
        self.assertEqual(result.evidence[0].source_url, doc.source_url)
        self.assertEqual(result.evidence[0].claim, doc.content)

    def test_documents_and_per_call_state_not_mutated(self):
        original = self.doc.model_dump()
        model = MockProductTechModel()
        retriever = MockProductTechRetriever([self.doc])
        agent = ProductTechAgent(model, retriever)
        agent.invoke(self.request)
        retriever.documents = []
        self.assertEqual(agent.invoke(self.request).evidence, [])
        self.assertEqual(self.doc.model_dump(), original)


class GraphAdapterTests(unittest.TestCase):
    def setUp(self):
        from agents import tech_brief, investment_evaluator, report_generator
        self.tech, self.investment, self.report = tech_brief, investment_evaluator, report_generator
        # 단위 테스트는 로컬 .env나 실제 API를 사용하지 않는다.
        for module in (self.tech, self.investment, self.report):
            factory_patch = patch.object(module, "MODEL_FACTORY", None)
            factory_patch.start()
            self.addCleanup(factory_patch.stop)
        data = mock_input()
        self.state = {
            "company_context": {"company_name": data["company_name"], "company_candidates": [data["company_name"]], "company_index": 0},
            "retry_state": {"retry_count": 0}, "evaluation_request": "연동 테스트",
            "evaluation_status": {"decision": "pending", "stop_reason": "in_progress"},
            "company_results": [], "errors": [], "route_history": [],
        }
        for field in ("founder_result", "market_result", "tech_result"):
            self.state[field] = specialist_to_graph(data[field], self.state)
        self.state["evidence_review"] = {
            "company_name": data["company_name"], "round_no": 1, "evidence_sufficient": True,
            "integrated_summary": "테스트 통합 근거", "reason": "테스트 검토", "missing_items": [], "conflicts": [],
        }
        self.doc = RetrievedDocument(document_id="test-doc", chunk_id="test-chunk",
                                     content="연동 테스트 원문", source_title="테스트.pdf", page=1,
                                     metadata={"provenance": "independent"})

    def prepare_report_state(self, decision="recommend"):
        self.state["investment_result"] = {
            "company_name": self.state["company_context"]["company_name"], "round_no": 1,
            "total_score": 100.0, "criteria_met": decision == "recommend", "decision_reason": "테스트 판단 이유",
        }
        self.state["evaluation_status"]["decision"] = decision

    def test_tech_run_converts_state_and_preserves_round(self):
        self.state["retry_state"]["retry_count"] = 1
        before = deepcopy(self.state)
        result = self.tech.run(self.state, model=MockProductTechModel(),
                               retriever=MockProductTechRetriever([self.doc]),
                               criteria=list(self.tech.TECH_CRITERIA.items()), ignored_option=True)
        self.assertEqual(result["round_no"], 2)
        self.assertEqual(result["agent"], "tech_brief")
        self.assertEqual(result["score"], 25)
        self.assertEqual(result["evidence"][0]["source"], "test-doc")
        self.assertEqual(self.state, before)

    def test_partial_scores_do_not_send_none_to_graph(self):
        result = self.tech.run(self.state, model=MockProductTechModel(), retriever=MockProductTechRetriever([]))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["scores"], [])
        self.assertNotIn("score", result)

    def test_investment_returns_string_and_uses_graph_decision(self):
        model = FakeModel("테스트 설명")
        before = deepcopy(self.state)
        output = self.investment.run(self.state, model=model, total_score=100, criteria_met=True, extra=True)
        self.assertIsInstance(output, str)
        self.assertIn("테스트 설명", output)
        payload = json.loads(model.messages[1]["content"])
        self.assertEqual(payload["investment_result"]["decision"], "recommend")
        self.assertTrue(payload["criteria_met"])
        self.assertEqual(self.state, before)

    def test_wrong_total_or_stale_result_rejected_before_model(self):
        model = FakeModel("사용되면 안 됨")
        with self.assertRaises(InputValidationError):
            self.investment.run(self.state, model=model, total_score=90, criteria_met=True)
        self.state["tech_result"]["round_no"] = 2
        with self.assertRaises(InputValidationError):
            self.investment.run(self.state, model=model, total_score=100, criteria_met=True)
        self.assertIsNone(model.messages)

    def test_report_preserves_graph_decisions_and_formats_markdown(self):
        for decision in ("recommend", "reject", "hold", "pending"):
            with self.subTest(decision=decision):
                self.prepare_report_state(decision)
                report = self.report.run(self.state, extra=True)
                self.assertIsInstance(report, str)
                self.assertIn("# SUMMARY", report)
                self.assertIn("## REFERENCE", report)
                self.assertIn(f"판단: {decision}", report)
                self.assertIn("100.0", report)

    def test_report_without_investment_or_specialists(self):
        self.state["evaluation_status"]["decision"] = "hold"
        del self.state["tech_result"]
        report = self.report.run(self.state)
        self.assertIn("미산정", report)
        self.assertIn("hold", report)
        self.assertIn("tech_result: 현재 회차 결과 없음", report)

    def test_saved_company_results_do_not_mix_current_company(self):
        self.prepare_report_state()
        record = {key: deepcopy(self.state[key]) for key in (
            "founder_result", "market_result", "tech_result", "evidence_review", "investment_result", "evaluation_status")}
        record.update(company_name=self.state["company_context"]["company_name"], round_no=1)
        self.state["company_results"] = [record, {
            "company_name": "자료 없는 테스트 기업", "round_no": 2,
            "evaluation_status": {"decision": "hold", "stop_reason": "insufficient_evidence"},
        }]
        self.state["company_context"]["company_name"] = "현재 상태의 다른 기업"
        before = deepcopy(self.state)
        report = self.report.run(self.state)
        self.assertIn("자료 없는 테스트 기업", report)
        self.assertIn("연동 테스트 기업", report)
        self.assertNotIn("현재 상태의 다른 기업", report)
        self.assertEqual(self.state, before)

    def test_top_level_model_settings_and_injection(self):
        from unittest.mock import Mock
        model = FakeModel("설정된 모델 설명")
        factory = Mock(return_value=model)
        with patch.object(self.investment, "MODEL_FACTORY", factory), \
             patch.object(self.investment, "MODEL_SETTINGS", {"model": "mock-model"}):
            self.investment.run(self.state, total_score=100, criteria_met=True)
            factory.assert_called_once_with(model="mock-model")
            factory.reset_mock()
            self.investment.run(self.state, model=FakeModel("외부 모델"), total_score=100, criteria_met=True)
            factory.assert_not_called()

    def test_actual_graph_dispatch_and_nodes(self):
        try:
            from graph import graph
        except ModuleNotFoundError as exc:
            if exc.name in ("langgraph", "graph", "graph.graph"):
                self.skipTest("Graph 및 LangGraph 설치 시 실행하는 통합 검사")
            raise
        with patch.object(self.tech, "MODEL", MockProductTechModel()), \
             patch.object(self.tech, "RETRIEVER", MockProductTechRetriever([self.doc])):
            result = graph.tech_brief(self.state)
        self.assertNotIn("errors", result)
        graph.validate_agent_result(result["tech_result"], self.state, "tech_brief")
        with patch.object(self.tech, "MODEL", MockProductTechModel()), \
             patch.object(self.tech, "RETRIEVER", MockProductTechRetriever([])):
            partial = graph.tech_brief(self.state)
        graph.validate_agent_result(partial["tech_result"], self.state, "tech_brief")
        self.assertEqual(partial["errors"][0]["code"], "AGENT_INCOMPLETE")
        with patch.object(self.investment, "MODEL", FakeModel("기준 충족 설명")):
            update = graph.investment_evaluator(self.state)
        self.assertNotIn("errors", update)
        self.state.update(update)
        self.state.update(graph.investment_gate(self.state))
        report_update = graph.report_generator(self.state)
        self.assertNotIn("errors", report_update)
        self.assertIn("recommend", report_update["report"])


class EnvironmentModelTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        dotenv = patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": "test-only-key", "LLM_MODEL": "test-model"})
        self.values = dotenv.start()
        self.addCleanup(dotenv.stop)
        constructor = patch("langchain_openai.ChatOpenAI")
        self.constructor = constructor.start()
        self.addCleanup(constructor.stop)

    def test_reads_project_env_and_constructs_without_request(self):
        from agents.evaluation_support import ENV_FILE
        model = create_model_from_env()
        self.values.assert_called_once_with(ENV_FILE)
        self.constructor.assert_called_once_with(model="test-model", api_key="test-only-key", timeout=60, max_retries=2)
        model.invoke.assert_not_called()
        self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_process_environment_wins(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "process-test-key", "LLM_MODEL": "process-model"}):
            create_model_from_env()
        self.assertEqual(self.constructor.call_args.kwargs["model"], "process-model")
        self.assertEqual(self.constructor.call_args.kwargs["api_key"], "process-test-key")

    def test_alias_and_conflict(self):
        self.values.return_value = {"OPENAI_API_KEY": "test-only-key", "OPENAI_MODEL": "alias-model"}
        create_model_from_env()
        self.assertEqual(self.constructor.call_args.kwargs["model"], "alias-model")
        self.values.return_value["LLM_MODEL"] = "different-model"
        with self.assertRaises(ModelConfigurationError):
            create_model_from_env()

    def test_missing_settings_fail_without_exposing_secret(self):
        for values in ({}, {"LLM_MODEL": "test-model"}, {"OPENAI_API_KEY": "test-only-key"}):
            with self.subTest(keys=list(values)):
                self.values.return_value = values
                with self.assertRaises(ModelConfigurationError) as caught:
                    create_model_from_env()
                self.assertNotIn("test-only-key", str(caught.exception))
        self.constructor.assert_not_called()

    def test_settings_can_override_model_without_hardcoded_logic(self):
        create_model_from_env(model="custom-test-model", temperature=0.2, timeout=20, max_retries=0)
        self.assertEqual(self.constructor.call_args.kwargs["model"], "custom-test-model")
        self.assertEqual(self.constructor.call_args.kwargs["temperature"], 0.2)
        self.assertEqual(self.constructor.call_args.kwargs["timeout"], 20)

    def test_all_three_module_defaults_use_env_factory(self):
        from agents import tech_brief, investment_evaluator, report_generator
        from agents.evaluation_support import resolve_model
        for module in (tech_brief, investment_evaluator, report_generator):
            with self.subTest(module=module.__name__):
                self.assertIs(module.MODEL_FACTORY, create_model_from_env)
                self.assertIs(resolve_model(None, module.MODEL, module.MODEL_FACTORY, module.MODEL_SETTINGS),
                              self.constructor.return_value)
                self.constructor.reset_mock()
                injected = FakeModel("외부 모델")
                self.assertIs(resolve_model(injected, module.MODEL, module.MODEL_FACTORY, module.MODEL_SETTINGS), injected)
                self.constructor.assert_not_called()

    def test_demo_default_remains_offline(self):
        import demo_agents
        with patch.object(sys, "argv", ["demo_agents.py"]), patch("sys.stdout", new_callable=StringIO) as stdout:
            demo_agents.main()
        self.assertEqual(json.loads(stdout.getvalue())["investment_result"]["decision"], "pending")
        self.values.assert_not_called()
        self.constructor.assert_not_called()

    def test_demo_check_config_never_invokes_model(self):
        import demo_agents
        with patch.object(sys, "argv", ["demo_agents.py", "--check-config"]), patch("sys.stdout", new_callable=StringIO) as stdout:
            demo_agents.main()
        self.assertIn("설정 확인 완료", stdout.getvalue())
        self.assertNotIn("test-only-key", stdout.getvalue())
        self.constructor.return_value.invoke.assert_not_called()

    def test_demo_llm_passes_created_model_to_both_agents(self):
        import demo_agents
        model = self.constructor.return_value
        data = demo_agents.build_mock_input()
        result = InvestmentEvaluator().invoke(data)
        report = ReportGenerator().invoke(dict(data, investment_result=result))
        with patch.object(demo_agents, "InvestmentEvaluator") as investment, \
             patch.object(demo_agents, "ReportGenerator") as reports, \
             patch.object(sys, "argv", ["demo_agents.py", "--llm"]), \
             patch("sys.stdout", new_callable=StringIO):
            investment.return_value.invoke.return_value = result
            reports.return_value.invoke.return_value = report
            demo_agents.main()
            investment.assert_called_once_with(model=model)
            reports.assert_called_once_with(model=model)


if __name__ == "__main__":
    unittest.main()
