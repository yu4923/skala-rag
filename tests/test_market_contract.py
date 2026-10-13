"""외부 API 없이 시장성 State 계약과 Graph 실행 제한을 검증한다."""
import json
import os
import unittest
from unittest.mock import Mock, patch

from agents import market_scout as market
from agents.evaluation_support import AgentResult, graph_agent_input
from graph import graph

# 그래프 import가 .env를 로드하므로 테스트에서는 외부 추적 전송을 끈다.
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'


def result_for(company, missing=None):
    request = market.MarketAgentInput(company_name=company, evaluation_request='시장성 평가')
    return market.MarketScoutAgent._unavailable(request, missing or ['계약 확인 필요'])


class MarketContractTests(unittest.TestCase):
    def setUp(self):
        self.state = graph.make_initial_state(['기업A', '기업B'], '시장성 평가')
        self.state['company_context'].update(company_description='전력 관리 서비스', target_market='국내 산업용 전력 시장')

    def test_context_and_market_round_are_authoritative(self):
        self.state['retry_state']['retry_count'] = 9
        request = market.input_from_state(self.state)
        self.assertEqual(request.round_no, 1)
        self.assertEqual(request.company_description, '전력 관리 서비스')
        self.assertEqual(request.target_market, '국내 산업용 전력 시장')
        self.assertEqual(request.evaluation_request, '시장성 평가')

    def test_missing_market_is_not_invented(self):
        self.state['company_context'].pop('target_market')
        self.assertEqual(market.input_from_state(self.state).target_market, '')

    def test_retry_passes_previous_missing_information(self):
        self.state['market_result'] = result_for('기업A')
        self.state['retry_state']['market_retry_count'] = 1
        request = market.input_from_state(self.state)
        self.assertEqual(request.round_no, 2)
        self.assertEqual(request.previous_missing_items, ['계약 확인 필요'])

    def test_previous_company_result_rejected(self):
        self.state['market_result'] = result_for('기업B')
        self.state['retry_state']['market_retry_count'] = 1
        with self.assertRaises(ValueError):
            market.input_from_state(self.state)

    def test_invalid_retry_counts_rejected_before_run(self):
        for value in (-1, 2, True, '1'):
            self.state['retry_state']['market_retry_count'] = value
            with self.assertRaises(ValueError):
                market.input_from_state(self.state)

    def test_as_node_and_module_run_share_contract_and_trace(self):
        agent = market.MarketScoutAgent(Mock(), lambda q: [], lambda q: [])
        error = market.MarketError(company='기업A', round=1, node='market.web_search', error_type='TimeoutError', message='웹 검색 실패')
        expected = market.MarketRun(result=result_for('기업A'), errors=[error], retrieval_queries=['pdf: 시장 규모'])
        with patch.object(agent, 'run', return_value=expected) as run_mock:
            with self.assertLogs(market.logger, level='INFO') as captured:
                direct = market.run(self.state, agent=agent, config={'metadata': {'caller': 'test'}})
                node = agent.as_node()(self.state)
            self.assertEqual(direct, node)
            self.assertIs(direct['market_result'], expected.result)
            self.assertEqual(direct['market_errors'][0]['error_type'], 'TimeoutError')
            self.assertEqual(direct['market_round_no'], 1)
            config = run_mock.call_args_list[0].kwargs['config']
            self.assertEqual(config['metadata']['trace_id'], self.state['trace_id'])
            self.assertEqual(config['metadata']['caller'], 'test')
            record = json.loads(captured.records[0].getMessage())
            self.assertEqual(record['trace_id'], self.state['trace_id'])
            self.assertEqual(record['errors'], direct['market_errors'])
            self.assertEqual(record['retrieval_queries'], ['pdf: 시장 규모'])

    def test_execution_failure_preserves_safe_error_and_result(self):
        agent = Mock()
        agent.run.side_effect = RuntimeError('secret-token')
        with self.assertLogs(market.logger, level='INFO') as captured:
            update = market.run(self.state, agent=agent)
        self.assertTrue(update['market_result'].needs_more_information)
        self.assertEqual(update['market_errors'][0]['error_type'], 'RuntimeError')
        self.assertNotIn('secret-token', captured.records[0].getMessage())

    def test_latest_run_replaces_previous_errors(self):
        self.state['market_errors'] = [{'error_type': 'OldError'}]
        agent = Mock()
        agent.run.return_value = market.MarketRun(result=result_for('기업A'), errors=[])
        self.assertEqual(market.run(self.state, agent=agent)['market_errors'], [])

    def test_graph_adapter_preserves_uncopyable_callback_context(self):
        from threading import Lock
        config = {'configurable': {'runtime_lock': Lock()}}
        agent = Mock()
        agent.run.return_value = market.MarketRun(result=result_for('기업A'))
        with patch.object(market, 'create_market_scout', return_value=agent):
            output = graph.market_scout(self.state, config=config)
        self.assertIsInstance(output['market_result'], AgentResult)
        passed = agent.run.call_args.kwargs['config']
        self.assertIs(passed['configurable']['runtime_lock'], config['configurable']['runtime_lock'])

    def test_graph_blocks_after_second_round(self):
        for count, last_round in ((1, 2), (2, 0)):
            self.state['retry_state']['market_retry_count'] = count
            self.state['market_round_no'] = last_round
            with patch.object(graph, 'call_agent') as call:
                self.assertEqual(graph.market_scout(self.state), {})
                call.assert_not_called()

    def test_next_company_resets_market_state(self):
        self.state.update(market_result=result_for('기업A'), market_errors=[{'message': 'old'}], market_round_no=2)
        self.state['retry_state']['market_retry_count'] = 1
        update = graph.next_company(self.state)
        self.assertEqual(update['company_context']['current_index'], 1)
        self.assertEqual(update['company_context']['company_name'], '기업B')
        self.assertEqual(update['company_context']['target_market'], '')
        self.assertIsNone(update['market_result'])
        self.assertEqual(update['market_errors'], [])
        self.assertEqual(update['retry_state']['market_retry_count'], 0)
        self.assertEqual(update['market_round_no'], 0)
        self.assertNotIn('trace_id', update)

    def test_search_error_does_not_invalidate_supported_assessment(self):
        agent = market.MarketScoutAgent(Mock(), lambda q: [], lambda q: [])
        source = market.MarketSource(source_title='시장 자료', source_type='pdf', source='market.pdf', page=1, excerpt='시장 성장과 유료 고객 계약이 확인된다.')
        def fake_create_agent(**kwargs):
            def invoke(payload, config):
                data = json.loads(payload['messages'][0][1])
                item = data['initial_evidence'][0][0]
                eid = item['evidence_id']
                assessment = market.MarketAssessment(
                    company_name=data['company_name'], round_no=data['round_no'], agent='market_scout',
                    status='success', summary='자료 범위 내 평가', missing_items=[],
                    evidence=[market.MarketCitation(evidence_id=eid, claim=source.excerpt, excerpt=source.excerpt, source_type='pdf', source='market.pdf', page=1)],
                    scores=[market.MarketScore(criterion=c, score=20, max_score=m, reason='제공 자료 기준', evidence_ids=[eid]) for c, m in market.MARKET_CRITERIA.items()], score=40,
                )
                return {'structured_response': assessment}
            return Mock(invoke=invoke)
        def failed_web(query):
            raise TimeoutError('sensitive exception should not be logged')
        agent.web_search = failed_web
        agent.rag_search = lambda query: [source]
        with patch.object(market, 'create_agent', side_effect=fake_create_agent):
            output = agent.run(market.input_from_state(self.state))
        self.assertEqual(len(output.errors), 1)
        self.assertFalse(output.result.needs_more_information)
        self.assertEqual(output.result.missing_information, [])
        self.assertEqual([v.score for v in output.result.evaluations], [20, 20])

    def test_step_limit_blocks_before_handler(self):
        self.state['step_count'] = self.state['max_steps']
        with patch.object(graph, 'call_agent') as call:
            with self.assertRaises(RuntimeError):
                graph.bounded_node(graph.market_scout)(self.state)
            call.assert_not_called()

    def test_full_graph_evaluates_ten_companies_with_market_retries(self):
        state = graph.make_initial_state([f'기업{i}' for i in range(10)], '시장성 평가')
        calls = []
        def fake_call(agent, state, **kwargs):
            company, round_no = graph.identity(state)
            if agent == 'market_scout':
                request = market.input_from_state(state)
                calls.append((company, request.round_no, request.previous_missing_items))
                return {'market_result': result_for(company), 'market_errors': [], 'market_round_no': request.round_no}
            if agent == 'investment_evaluator':
                graph_agent_input(state)  # downstream 공통 결과 수신 계약도 검증
                return {'company_name': company, 'round_no': round_no, 'total_score': 50.0,
                        'criteria_met': company == '기업0', 'provisional': True, 'decision_reason': '테스트 판단'}
            if agent == 'report_generator':
                graph_agent_input(state)
                return '테스트 보고서'
            criteria = graph.CRITERIA[agent]
            return {'company_name': company, 'round_no': round_no, 'agent': agent, 'status': 'success',
                    'summary': '테스트 평가', 'missing_items': [],
                    'evidence': [{'evidence_id': agent, 'claim': '자료', 'excerpt': '자료', 'source_type': 'input', 'source': 'test'}],
                    'scores': [{'criterion': c, 'score': maximum, 'max_score': maximum, 'reason': '자료 기준', 'evidence_ids': [agent]} for c, maximum in criteria],
                    'score': sum(m for _, m in criteria)}
        with patch.object(graph, 'call_agent', side_effect=fake_call):
            output = graph.build_graph().invoke(state, config={'recursion_limit': 500})
        self.assertEqual(len(output['company_results']), 10)
        self.assertEqual(len(calls), 19)
        self.assertEqual(calls[0], ('기업0', 1, []))
        self.assertEqual(calls[1], ('기업1', 1, []))
        self.assertEqual(calls[2], ('기업1', 2, ['계약 확인 필요']))
        self.assertEqual(output['company_context']['current_index'], 9)
        self.assertEqual(output['trace_id'], state['trace_id'])
        self.assertGreater(output['step_count'], 10)
        self.assertLessEqual(output['step_count'], output['max_steps'])
        self.assertIsInstance(output['market_result'], AgentResult)
        self.assertIsNone(output['market_result'].evaluations[0].score)
        self.assertEqual(output['report'], '테스트 보고서')


if __name__ == '__main__':
    unittest.main()
