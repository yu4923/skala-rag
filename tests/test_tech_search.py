"""제품 요약 Agent의 원문·페이지·예외 반환 계약을 검증한다."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from rag.tech_ingest import INDEX, PDF
from rag.tech_search import document_to_result, create_rag_search


def sample_doc(**changes):
    """테스트에 필요한 명시적인 문서 메타데이터를 구성한다."""
    metadata = dict(source_title='실제 문서명', source='sample.pdf', page_number=46,
                    company_id='LC', content_kind='body')
    metadata.update(changes)
    return Document(page_content='실제 원문\n[LC2]', metadata=metadata)


def write_test_manifest(directory):
    """FAISS를 mock하는 테스트에 필요한 최소 인덱스 계약을 저장한다."""
    manifest = {
        'schema_version': 2,
        'model': 'intfloat/multilingual-e5-small',
        'normalized': True,
        'distance': 'squared_l2',
        'source_sha256': hashlib.sha256(PDF.read_bytes()).hexdigest(),
        'dimension': 384,
        'companies': {'LC': 13},
    }
    Path(directory, 'manifest.json').write_text(json.dumps(manifest))


class TechSearchTests(unittest.TestCase):
    """반환값의 출처 정확성과 장애·무결과의 구분을 검사한다."""

    def test_original_text_and_page(self):
        """실제 원문과 1-based 번호를 수정하지 않는다."""
        doc = sample_doc(page_number=49, page=48)
        result = document_to_result(doc)
        self.assertEqual(result['excerpt'], doc.page_content)
        self.assertEqual(result['page'], 49)
        self.assertEqual(result['source_title'], '실제 문서명')
        self.assertEqual(result['source'], 'sample.pdf')
        self.assertEqual(set(result), {'source_title', 'source_type', 'source',
                                      'excerpt', 'page', 'published_date'})
        self.assertIsNone(result['published_date'])

    def test_page_bases(self):
        """번호 기준이 선언된 경우에만 페이지를 변환한다."""
        self.assertEqual(document_to_result(sample_doc(page_number=None, page=0,
                                                       page_index_base=0))['page'], 1)
        self.assertEqual(document_to_result(sample_doc(page_number=None, page=12,
                                                       page_index_base=1))['page'], 12)
        with self.assertRaises(ValueError):
            document_to_result(sample_doc(page_number=None, page=0))

    def test_required_metadata_and_dates(self):
        """필수 출처 누락은 오류이며 날짜·버전을 추정하지 않는다."""
        for field in ['source_title', 'source', 'page_number']:
            with self.assertRaises(ValueError):
                document_to_result(sample_doc(**{field: None}))
        self.assertEqual(document_to_result(sample_doc(published_date='2026-06-01'))
                         ['published_date'], '2026-06-01')
        with self.assertRaises(ValueError):
            document_to_result(sample_doc(published_date='2026-02-30'))
        self.assertIsNone(document_to_result(sample_doc(document_version='v2'))['published_date'])

    def test_empty_and_missing_index(self):
        """빈 질문과 저장소 장애를 서로 다른 결과로 처리한다."""
        if (INDEX / 'manifest.json').is_file():
            self.assertEqual(create_rag_search('LC')(' '), [])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                create_rag_search('LC', index_dir=directory)

    def test_duplicates_threshold_and_failure(self):
        """중복 제거·임계값 미달·검색 장애 전파를 검사한다."""
        store = MagicMock()
        store.index.ntotal, store.index.d = 13, 384
        doc = sample_doc()
        store.similarity_search_with_score.return_value = [(doc, 0.2), (doc, 0.3)]
        with tempfile.TemporaryDirectory() as directory:
            write_test_manifest(directory)
            with patch('rag.tech_search.FAISS.load_local', return_value=store):
                search = create_rag_search('LC', index_dir=directory)
                self.assertEqual(len(search('기술')), 1)
                strict = create_rag_search('LC', min_similarity=0.99,
                                           index_dir=directory)
                self.assertEqual(strict('기술'), [])
                store.similarity_search_with_score.side_effect = RuntimeError('검색 장애')
                with self.assertRaises(RuntimeError):
                    search('기술')

    def test_body_preference_preserves_relevance(self):
        """근접한 결과에서 본문을 우선하되 무관한 본문을 끌어올리지 않는다."""
        store = MagicMock()
        store.index.ntotal, store.index.d = 13, 384
        reference, body = sample_doc(content_kind='references'), sample_doc(page_number=47)
        with tempfile.TemporaryDirectory() as directory:
            write_test_manifest(directory)
            with patch('rag.tech_search.FAISS.load_local', return_value=store):
                search = create_rag_search('LC', top_k=1, index_dir=directory)
                store.similarity_search_with_score.return_value = [(reference, 0.2),
                                                                    (body, 0.21)]
                self.assertEqual(search('기능')[0]['page'], 47)
                store.similarity_search_with_score.return_value = [(reference, 0.2),
                                                                    (body, 0.8)]
                self.assertEqual(search('기능')[0]['page'], 46)


if __name__ == '__main__':
    unittest.main()
