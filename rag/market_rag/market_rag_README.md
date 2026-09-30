# Market RAG 연결 방법

## 폴더 구성

```text
rag/
├── data/
│   ├── market/
│   │   └── 시장성_종합자료2.pdf
│   └── tech/
│       └── tech-10-startups-50pages-2026-09-30.pdf
├── indexes/
│   ├── market/
│   └── product/
└── market_rag/
    ├── __init__.py
    ├── market_scout.py
    ├── market_requirements.txt
    └── market_rag_README.md
```

`rag/indexes/market`에는 최초 실행 시 `market.faiss`, `market_chunks.json`, `market_manifest.json`이 자동 생성된다. 이 파일들은 PDF에서 다시 만들 수 있으므로 Git에는 포함하지 않는다.

## 설치

프로젝트 가상환경에서 다음 명령을 실행한다.

```bash
python -m pip install -r rag/market_rag/market_requirements.txt
```

`intfloat/multilingual-e5-small` 모델은 최초 초기화 시 내려받아 로컬 모델 캐시에 저장한다.

## import와 초기화

애플리케이션 시작 시 한 번 초기화한다.

```python
from rag.market_rag import initialize_market_rag, rag_search

initialize_market_rag()
```

PDF가 변경됐거나 인덱스를 강제로 다시 만들 때는 다음과 같이 실행한다.

```python
initialize_market_rag(force_rebuild=True)
```

## 검색

```python
results = rag_search("국내 ESS 시장 성장률과 설치 장벽")
```

반환 예시:

```python
[
    {
        "source_title": "시장성_종합자료2",
        "source_type": "pdf",
        "source": "rag/data/market/시장성_종합자료2.pdf",
        "excerpt": "PDF에서 검색된 실제 원문",
        "page": 17,
        "published_date": None,
    }
]
```

검색 결과는 관련도순 최대 3개다. 결과가 없거나 검색어가 비어 있으면 `[]`를 반환한다. PDF, 임베딩 모델 또는 인덱스에 문제가 있으면 `MarketRAGError`를 발생시킨다.

기업별 PoC, 유료 고객, 반복 구매, 장기계약의 웹 검색과 시장성 점수 계산은 호출하는 시장성 검증 Agent가 담당한다.

## 설정값

| 환경변수 | 기본값 | 설명 |
|---|---:|---|
| `MARKET_RAG_DATA_DIR` | `rag/data/market` | 검색할 PDF 폴더 |
| `MARKET_RAG_INDEX_DIR` | `rag/indexes/market` | FAISS 캐시 폴더 |
| `MARKET_RAG_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | 임베딩 모델 |
| `MARKET_RAG_CHUNK_SIZE` | `440` | 청크 최대 토큰 수 |
| `MARKET_RAG_CHUNK_OVERLAP` | `70` | 청크 중첩 토큰 수 |
| `MARKET_RAG_CANDIDATE_K` | `10` | 검색기별 후보 수 |
| `MARKET_RAG_RESULT_K` | `3` | 최종 반환 수 |
| `MARKET_RAG_DENSE_WEIGHT` | `0.6` | FAISS 순위 결합 가중치 |
| `MARKET_RAG_SPARSE_WEIGHT` | `0.4` | BM25 순위 결합 가중치 |
| `MARKET_RAG_MIN_DENSE_SCORE` | `0.85` | 의미 검색 최소 유사도 |
| `MARKET_RAG_PUBLISHED_DATE` | 미설정 | 확인된 경우 `YYYY-MM-DD`로 지정 |

macOS에서 네이티브 라이브러리 충돌로 종료 코드 139가 발생하면 `OMP_NUM_THREADS=1`과 `TOKENIZERS_PARALLELISM=false`를 설정한 뒤 다시 실행한다.
