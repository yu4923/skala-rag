"""모의 모델·검색으로 실제 LangChain 도구 루프 및 팀 결과 통합을 검증한다."""
import json
import math
import unittest
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, ValidationError

from agents.evaluation_support import CRITERIA, InvestmentAgentInput
from agents.investment_evaluator import InvestmentEvaluator
from agents.market_scout import (
    MarketAgentInput, MarketAssessment, MarketSource, calculate_growth, create_market_scout,
)
from prompts.market_scout_prompt import MARKET_SCOUT_PROMPT


def pdf_source():
    return MarketSource(source_title="시장 보고서", source_type="pdf", source="market.pdf", page=7,
                        excerpt="국내 가정용 시장은 2024년 100억원, 2026년 121억원이다.")


def web_source():
    return MarketSource(source_title="고객 계약 공시", source_type="web", source="https://example.org/contracts",
                        excerpt="평가 기업은 유료 고객 100가구와 2년 계약을 체결했다.")


def request(company="테스트 에너지", **kwargs):
    return MarketAgentInput(company_name=company, evaluation_request="국내 시장과 실제 고객 수요 평가", **kwargs)


class ScriptedMarketModel(BaseChatModel):
    mutation: str = ""
    followup: bool = False
    growth: bool = False
    names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted-market-test"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"names": [convert_to_openai_tool(item)["function"]["name"] for item in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        assert "MarketAssessment" in self.names
        payload = json.loads(next(message.content for message in messages if message.type == "human"))
        sources = list(payload["input_evidence"])
        for batch in payload["initial_evidence"]:
            if isinstance(batch, list):
                sources.extend(batch)
        tool_messages = [message for message in messages if message.type == "tool"]
        for message in tool_messages:
            if message.name in {"search_documents", "search_web"}:
                batch = json.loads(message.content)
                if isinstance(batch, list):
                    sources.extend(batch)
        if self.followup and not tool_messages:
            call = dict(name="search_documents", args={"query": "유료 고객 독립 확인"}, id="search")
        elif self.growth and not any(message.name == "calculate_growth" for message in tool_messages):
            call = dict(name="calculate_growth", args=dict(start_value=100, end_value=121,
                        start_year=2024, end_year=2026, unit="억원", geography="국내",
                        market_scope="가정용 시장", evidence_ids=[sources[0]["evidence_id"]]), id="growth")
        else:
            citations = {}
            scores = []
            for index, (criterion, maximum) in enumerate(payload["criteria"].items()):
                desired = "pdf" if index == 0 else "web"
                item = next((source for source in sources if source["source_type"] == desired), sources[0])
                eid = item["evidence_id"]
                citations[eid] = dict(evidence_id=eid, claim="확인한 자료에 따른 평가",
                                     excerpt=item["excerpt"], source_type=item["source_type"],
                                     source=item["source"], page=item["page"])
                scores.append(dict(criterion=criterion, score=20 if index == 0 else 15,
                                   max_score=maximum, reason="출처와 원문에 근거한 평가", evidence_ids=[eid]))
            output = dict(company_name=payload["company_name"], round_no=payload["round_no"],
                          agent="market_scout", status="success", summary="시장 확대와 유료 고객 확인",
                          evidence=list(citations.values()), missing_items=[], scores=scores, score=35)
            if self.mutation == "unknown_id":
                original = output["evidence"][0]["evidence_id"]
                output["evidence"][0]["evidence_id"] = "invented"
                for score in scores:
                    score["evidence_ids"] = ["invented" if eid == original else eid for eid in score["evidence_ids"]]
            elif self.mutation == "quote":
                output["evidence"][0]["excerpt"] = "존재하지 않는 원문"
            elif self.mutation == "page":
                output["evidence"][0]["page"] = 99
            elif self.mutation == "source":
                output["evidence"][0]["source"] = "invented.pdf"
            elif self.mutation == "company":
                output["company_name"] = "다른 기업"
            elif self.mutation == "round":
                output["round_no"] = 2
            elif self.mutation == "total":
                output["score"] = 50
            elif self.mutation == "duplicate":
                scores[1]["criterion"] = scores[0]["criterion"]
            elif self.mutation == "partial":
                scores[1]["score"], scores[1]["evidence_ids"] = None, []
                output.update(status="partial", score=None, missing_items=["유료 계약 원문 확인 필요"])
            elif self.mutation == "error":
                output.update(status="error", missing_items=["대상 기업 식별 불가"])
            elif self.mutation == "decimal":
                scores[0]["score"], scores[1]["score"] = 12.1, 12.2
                output["score"] = 24.3
            call = dict(name="MarketAssessment", args=output, id="result")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


def agent(model=None, **kwargs):
    return create_market_scout(model or ScriptedMarketModel(),
                               web_search=lambda query: [web_source()],
                               rag_search=lambda query: [pdf_source()], **kwargs)


class MarketScoutTests(unittest.TestCase):
    def test_actual_tool_loop_and_growth(self):
        run = agent(ScriptedMarketModel(followup=True, growth=True)).run(request())
        self.assertFalse(run.errors)
        self.assertEqual(run.assessment.status, "success")
        self.assertEqual(run.assessment.score, 35)
        self.assertEqual([item.score for item in run.result.evaluations], [20, 15])
        self.assertEqual(run.result.agent_name, "market")
        self.assertFalse(run.result.needs_more_information)
        self.assertEqual(run.result.evidence[0].page, 7)
        self.assertIn("market.pdf", run.result.evidence[0].source_title)
        self.assertIn("원문:", run.result.evidence[0].claim)
        self.assertEqual(len(run.retrieval_queries), 3)
        self.assertAlmostEqual(run.calculations[0]["cagr_percent"], 10)
        self.assertEqual(run.calculations[0]["evidence_ids"], [run.result.evidence[0].evidence_id])

    def test_external_prompt_is_passed_without_formatting(self):
        self.assertEqual(agent().system_prompt, MARKET_SCOUT_PROMPT)
        custom = '별도 프롬프트: {"scores": [{"score": 25}]}'
        self.assertEqual(agent(system_prompt=custom).system_prompt, custom)
        with self.assertRaises(ValueError):
            agent(system_prompt=" ")

    def test_tam_customer_and_metadata_contract_checks(self):
        for mutation in ("unknown_id", "quote", "page", "source", "company", "round", "total", "duplicate"):
            with self.subTest(mutation=mutation):
                with self.assertLogs("agents.market_scout", level="WARNING"):
                    run = agent(ScriptedMarketModel(mutation=mutation), recursion_limit=5).run(request())
                self.assertTrue(run.errors)
                self.assertTrue(run.result.needs_more_information)
                self.assertTrue(all(item.score is None for item in run.result.evaluations))

    def test_partial_preserves_valid_score_and_only_used_sources(self):
        run = agent(ScriptedMarketModel(mutation="partial")).run(request())
        self.assertEqual(run.result.evaluations[0].score, 20)
        self.assertIsNone(run.result.evaluations[1].score)
        self.assertIsNone(run.assessment.score)
        self.assertEqual(len(run.result.evidence), 1)
        self.assertIn("유료 계약 원문 확인 필요", run.result.missing_information)
        self.assertEqual(run.result.confidence, "low")

    def test_error_status_never_leaks_scores(self):
        result = agent(ScriptedMarketModel(mutation="error")).invoke(request())
        self.assertTrue(all(item.score is None for item in result.evaluations))
        self.assertTrue(result.needs_more_information)

    def test_exact_decimal_sum(self):
        run = agent(ScriptedMarketModel(mutation="decimal")).run(request())
        self.assertFalse(run.errors)
        self.assertEqual(run.assessment.score, 24.3)

    def test_no_sources_returns_unscored_without_calling_model(self):
        scout = create_market_scout(object(), lambda query: [], lambda query: [])
        with self.assertLogs("agents.market_scout", level="WARNING"):
            run = scout.run(request())
        self.assertIsNone(run.assessment)
        self.assertEqual(run.result.evidence, [])
        self.assertEqual(run.errors[0].error_type, "LookupError")

    def test_failed_search_and_model_error_are_sanitized(self):
        def broken(query):
            raise RuntimeError("Authorization: SECRET")
        scout = create_market_scout(ScriptedMarketModel(), broken, lambda query: [pdf_source()])
        with self.assertLogs("agents.market_scout", level="WARNING") as logs:
            run = scout.run(request())
        self.assertIsNotNone(run.result.evaluations[0].score)
        self.assertTrue(run.result.needs_more_information)
        self.assertNotIn("SECRET", run.model_dump_json() + str(logs.output))
        with patch("agents.market_scout.create_agent", side_effect=TimeoutError("SECRET")):
            with self.assertLogs("agents.market_scout", level="WARNING"):
                run = agent().run(request())
        self.assertEqual(run.errors[0].error_type, "TimeoutError")
        self.assertNotIn("SECRET", run.model_dump_json())

    def test_budget_and_second_round_missing_items(self):
        calls = []
        def rag(query):
            calls.append(query)
            return [pdf_source()]
        scout = create_market_scout(ScriptedMarketModel(followup=True), lambda query: [web_source()], rag,
                                    max_searches=2)
        run = scout.run(request(round_no=2, previous_missing_items=["반복 구매 증빙"]))
        self.assertEqual(len(run.retrieval_queries), 2)
        self.assertEqual(len(calls), 1)
        self.assertIn("반복 구매 증빙", calls[0])
        self.assertEqual(run.assessment.round_no, 2)
        with self.assertRaises(ValidationError):
            request(round_no=3)

    def test_input_only_requires_independent_verification(self):
        source = MarketSource(source_title="기업 제출 자료", source_type="input", source="company-intro",
                              excerpt="기업이 주장하는 유료 고객은 100가구이다.")
        scout = create_market_scout(ScriptedMarketModel(), lambda query: [], lambda query: [])
        run = scout.run(request(input_evidence=[source]))
        self.assertTrue(run.result.needs_more_information)
        self.assertTrue(any("교차검증" in item for item in run.result.missing_information))
        self.assertEqual(len(run.result.evidence), 1)

    def test_source_validation_and_wrong_backend_type(self):
        with self.assertRaises(ValidationError):
            MarketSource(source_title="문서", source_type="pdf", source="x.pdf", excerpt="원문")
        with self.assertRaises(ValidationError):
            MarketSource(source_title="웹", source_type="web", source="file:///x", excerpt="원문")
        scout = create_market_scout(ScriptedMarketModel(), lambda query: [pdf_source()], lambda query: [pdf_source()])
        with self.assertLogs("agents.market_scout", level="WARNING"):
            run = scout.run(request())
        self.assertTrue(run.errors)
        self.assertTrue(run.result.needs_more_information)

    def test_input_validation_and_isolation(self):
        for kwargs in ({"company_name": " "}, {"criteria": {"임의 항목": 100}}, {"round_no": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                MarketAgentInput(evaluation_request="평가", **({"company_name": "기업"} | kwargs))
        scout = agent()
        first = scout.invoke(request("기업 A"))
        second = scout.invoke(request("기업 B"))
        self.assertFalse(set(item.evidence_id for item in first.evidence) & set(item.evidence_id for item in second.evidence))

    def test_integration_with_investment_evaluator_and_node(self):
        market = agent().invoke(request())
        data = dict(company_name=market.company_name, market_result=market)
        for field, role in (("founder_result", "founder"), ("tech_result", "technology")):
            data[field] = dict(agent_name=role, company_name=market.company_name, confidence="medium",
                needs_more_information=False, risks=[], missing_information=[], evaluations=[], evidence=[])
            for index, (criterion, maximum) in enumerate(CRITERIA[role].items()):
                eid = f"{role}-{index}"
                data[field]["evaluations"].append(dict(criterion=criterion, score=maximum, reason="테스트", evidence_ids=[eid]))
                data[field]["evidence"].append(dict(evidence_id=eid, claim="테스트", source_title="문서", page=1))
        investment = InvestmentEvaluator().invoke(InvestmentAgentInput(**data))
        self.assertEqual(investment.total_score, 85)
        self.assertEqual(investment.decision, "pending")
        node_result = agent().as_node()({"market_input": request().model_dump()})
        self.assertEqual(node_result["market_result"].agent_name, "market")
        self.assertEqual(node_result["errors"], [])

    def test_growth_rejects_invalid_numbers_and_handles_decline(self):
        for values in ((0, 100, 1), (100, 100, 0), (math.inf, 1, 1), (-1, 1, 1), (1, 1, math.nan)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                calculate_growth(*values)
        self.assertEqual(calculate_growth(100, 81, 2), {"growth_percent": -19, "cagr_percent": -10})


if __name__ == "__main__":
    unittest.main()
