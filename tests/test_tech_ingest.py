"""검색·LLM 호출 없이 PDF 분할과 저장된 벡터 DB를 검증한다."""
import json
import unittest
import faiss
import numpy as np
from rag.tech_ingest import COMPANIES, INDEX, CHUNK_SIZE, load_documents, split_documents


class TechBriefTests(unittest.TestCase):
    """회사·페이지 경계와 실제 저장된 벡터 수를 확인한다."""

    def test_page_boundaries(self):
        """10개 기업에 다섯 페이지씩 배정되는지 검사한다."""
        docs = load_documents()
        self.assertEqual([d.metadata['page_number'] for d in docs], list(range(1, 51)))
        for code in COMPANIES:
            self.assertEqual(sum(d.metadata['company_id'] == code for d in docs), 5)

    def test_chunk_boundaries(self):
        """각 청크가 원문 페이지 안에 있고 회사와 위치가 보존되는지 검사한다."""
        chunks = split_documents(load_documents())
        self.assertEqual(len({d.metadata['chunk_id'] for d in chunks}), len(chunks))
        for doc in chunks:
            meta = doc.metadata
            start = meta['start_index']
            self.assertLessEqual(len(doc.page_content), CHUNK_SIZE)
            self.assertGreaterEqual(start, 0)
            self.assertEqual(meta['page_context'][start:start + len(doc.page_content)],
                             doc.page_content)
            self.assertEqual(meta['company_id'], list(COMPANIES)[(meta['page_number'] - 1) // 5])

    @unittest.skipUnless((INDEX / "manifest.json").is_file(),
                         "생성 인덱스가 있는 환경에서 검사")
    def test_saved_indexes(self):
        """디스크에서 10개 FAISS를 다시 읽고 개수·차원·정규화를 검사한다."""
        manifest = json.loads((INDEX / 'manifest.json').read_text())
        expected = split_documents(load_documents())
        self.assertEqual(manifest['chunks'], len(expected))
        for code in COMPANIES:
            index = faiss.read_index(str(INDEX / code / 'index.faiss'))
            count = sum(d.metadata['company_id'] == code for d in expected)
            self.assertEqual(index.ntotal, count)
            self.assertEqual(index.ntotal, manifest['companies'][code])
            self.assertEqual(index.d, 384)
            self.assertTrue((INDEX / code / 'index.pkl').is_file())
            vectors = index.reconstruct_n(0, index.ntotal)
            self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
