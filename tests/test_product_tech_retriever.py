"""담당자의 검색 호출 규약과 반환 필드·기업 필터를 검증한다."""
import unittest
from unittest.mock import patch
from rag.product_tech_retriever import ProductTechRAGRetriever, resolve_company, to_agent_document


class ProductTechRetrieverTests(unittest.TestCase):
    """실제 검색 경계에 필요한 변환·중복·실패 전파를 검사한다."""

    def test_identity_and_company_alias(self):
        """별칭이 같은 DB를 선택하고 원문·페이지가 ID에 반영되는지 검사한다."""
        result = dict(source='tech.pdf', page=7, excerpt='원문\n[HZ2]',
                      source_title='기술 자료집', published_date=None)
        doc = to_agent_document(result, '해줌', 'HZ')
        self.assertEqual(doc['content'], result['excerpt'])
        self.assertIsNone(doc['source_url'])
        self.assertNotIn('evidence_id', doc)
        self.assertNotIn('retrieval_score', doc)
        self.assertEqual(doc['chunk_id'], to_agent_document(result, 'Haezoom', 'HZ')['chunk_id'])
        self.assertNotEqual(doc['chunk_id'], to_agent_document({**result, 'page': 8}, '해줌', 'HZ')['chunk_id'])
        self.assertEqual(resolve_company('인코어드테크놀로지스'), 'EC')
        self.assertEqual(resolve_company('Haezoom'), 'HZ')
        with self.assertRaises(ValueError):
            resolve_company('없는회사')

    def test_multi_query_and_error(self):
        """질의 중복·청크 중복을 제거하고 장애를 빈 결과로 숨기지 않는다."""
        result = dict(source='tech.pdf', page=7, excerpt='원문',
                      source_title='기술 자료집', published_date=None)
        with patch('rag.product_tech_retriever.create_rag_search') as factory:
            search = factory.return_value
            search.return_value = [result]
            retriever = ProductTechRAGRetriever()
            docs = retriever.search(company_name='해줌', queries=['기능', '기능', '입력'])
            self.assertEqual(len(docs), 1)
            self.assertEqual(search.call_count, 2)
            retriever.search(company_name='Haezoom', queries=['기능'])
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(retriever.search(company_name='해줌', queries=[]), [])
            search.side_effect = RuntimeError('검색 장애')
            with self.assertRaises(RuntimeError):
                retriever.search(company_name='해줌', queries=['기능'])
