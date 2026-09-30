# 제품·기술 Agent RAG 전달 안내

작성일: 2026-09-30

저장소: <https://github.com/yu4923/skala-rag>
기준 커밋: `d4e936d`

## 1. 담당 범위

제품·기술 Agent에 주입할 RAG 모듈을 구현했다. 담당자의
`agents/tech_brief.py`는 수정하지 않았다.

RAG 모듈이 담당하는 범위:

- 10개 기업, 총 50쪽 PDF 로딩
- 페이지와 기업 메타데이터 보존
- 문서 청크 분할
- `intfloat/multilingual-e5-small` 임베딩
- 기업별 FAISS 인덱스 생성과 검색
- 기업명 기반 인덱스 라우팅
- 복수 질의 처리와 동일 청크 중복 제거
- 담당자의 `ProductTechRetriever` 반환 계약으로 변환

RAG 모듈이 담당하지 않는 범위:

- LLM 호출과 기술 요약
- 기술력 평가와 점수 계산
- `evidence_id` 생성
- 근거 충분성·충돌 여부에 대한 최종 판단
- `agents/tech_brief.py` 구현

## 2. 파일 구성

```text
rag/
├── __init__.py
└── tech_rag/
    ├── __init__.py
    ├── product_tech_retriever.py  # ProductTechAgent 주입 객체
    ├── tech_search.py             # 기업별 FAISS 검색과 결과 변환
    ├── tech_ingest.py             # PDF 로딩·분할·벡터 DB 생성
    ├── tech_embeddings.py         # 로컬 E5 임베딩
    ├── requirements.txt
    └── requirements.lock.txt

rag/data/tech/
└── tech-10-startups-50pages-2026-09-30.pdf
```

상세 구현 계약은 `docs/integration/product-rag-handoff.md`에 있다.

## 3. 환경 구성

프로젝트 루트에서 실행한다.

```bash
cd skala-rag
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r rag/tech_rag/requirements.lock.txt
```

OpenAI API 키는 필요하지 않다. 임베딩은 로컬 E5 모델을 사용한다.
모델 캐시가 없는 환경에서는 최초 실행 시 Hugging Face에서 모델을
다운로드한다.

## 4. 벡터 DB 생성

FAISS 인덱스는 Git에 포함하지 않았다. 운영체제·FAISS 버전 차이와
pickle 역직렬화 위험을 줄이기 위해 각 실행 환경에서 PDF로 재생성한다.

```bash
python -m rag.tech_rag.tech_ingest
```

생성 위치:

```text
rag/indexes/product/
├── manifest.json
├── EN/
├── HZ/
├── SH/
├── SY/
├── EC/
├── VP/
├── VG/
├── EX/
├── CR/
└── LC/
```

각 기업 폴더에는 `index.faiss`와 `index.pkl`이 생성된다. 이미 같은 출력
경로가 존재하면 기존 인덱스를 보호하기 위해 `FileExistsError`가 발생한다.

직접 생성했거나 신뢰할 수 있는 경로로 전달받은 `index.pkl`만 사용한다.

## 5. Agent 연결

담당자 코드에 다음 객체를 주입한다.

```python
from agents.tech_brief import ProductTechAgent
from rag.tech_rag.product_tech_retriever import ProductTechRAGRetriever

retriever = ProductTechRAGRetriever(top_k=3)

agent = ProductTechAgent(
    model=model,
    retriever=retriever,
)
```

담당자의 Agent는 다음 형식으로 검색기를 호출할 수 있다.

```python
documents = retriever.search(
    company_name="해줌",
    queries=[
        "발전량 예측 API의 입력 데이터와 출력 결과",
        "주요 제품 기능과 사용 절차",
    ],
)
```

`ProductTechRAGRetriever`는 기업별 검색 객체를 최초 사용 시 생성하고 이후
재사용한다. 인덱스를 재구축했다면 Retriever 객체도 다시 생성한다.

## 6. 반환 데이터

담당자의 `RetrievedDocument` 계약에 맞춰 다음 형태로 반환한다.

```python
[
    {
        "document_id": "rag/data/tech/tech-10-startups-50pages-2026-09-30.pdf",
        "chunk_id": "문서·페이지·원문으로 만든 SHA-256",
        "content": "검색된 PDF 청크의 실제 원문",
        "source_title": "에너지 스타트업 기술 자료집",
        "page": 7,
        "source_url": None,
        "metadata": {
            "source_type": "pdf",
            "source": "rag/data/tech/tech-10-startups-50pages-2026-09-30.pdf",
            "company_id": "HZ",
            "company_name": "해줌",
            "canonical_company_name": "해줌",
            "published_date": None,
        },
    }
]
```

반환 규칙:

- `content`는 PDF에서 추출한 청크 원문이다.
- 문장을 요약·결합·재작성하지 않는다.
- `page`는 PDF 뷰어 기준 1부터 시작한다.
- `chunk_id`는 문서 경로·페이지·원문을 이용한 결정적 SHA-256이다.
- 같은 청크는 질의가 달라도 동일한 `chunk_id`를 갖는다.
- 로컬 PDF이므로 `source_url`은 `None`이다.
- 확인된 발행일이 없어 `published_date`는 `None`이다.
- `retrieval_score`, `evidence_id`, 평가 점수는 반환하지 않는다.
- `provenance="independent"`를 임의로 부여하지 않는다.

담당자의 `tech_brief.py`가 검색 문서로부터 `evidence_id`를 생성하고 인용과
평가 결과를 검증한다.

## 7. 기업 코드와 페이지

| 코드 | 기업 | PDF 페이지 |
|---|---|---:|
| `EN` | 엔라이튼 | 1–5 |
| `HZ` | 해줌 | 6–10 |
| `SH` | 식스티헤르츠 | 11–15 |
| `SY` | 시너지 | 16–20 |
| `EC` | 인코어드 | 21–25 |
| `VP` | 브이피피랩 | 26–30 |
| `VG` | 브이젠 | 31–35 |
| `EX` | 에너지엑스 | 36–40 |
| `CR` | 크로커스 | 41–45 |
| `LC` | 리셀 | 46–50 |

한국어 정식 명칭, 기업 코드와 일부 영문 별칭을 지원한다. 지원하지 않는
기업명은 `ValueError`로 처리한다.

## 8. 검색 동작

기본값은 검색어 하나당 최대 3개 청크다.

```python
retriever = ProductTechRAGRetriever(top_k=3)
```

검색어가 두 개라면 최대 6개에서 중복 청크를 제거한 결과를 반환한다.
입력한 검색어 순서와 각 검색어 내부의 검색 순서를 유지한다. 서로 다른
검색어 결과를 하나의 전역 점수로 다시 정렬하지 않는다.

빈 질의 목록 또는 공백만 있는 질의 목록은 `[]`를 반환한다. 중복 질의는
한 번만 검색한다. 하나의 질의에서 검색 장애가 발생하면 부분 성공을 정상
결과처럼 반환하지 않고 예외를 전달한다.

## 9. 청크와 임베딩 기준

- 임베딩 모델: `intfloat/multilingual-e5-small`
- 벡터 차원: 384
- 정규화: 적용
- 문서 접두사: `passage: `
- 질문 접두사: `query: `
- 청크 최대 길이: 800자
- 청크 중첩: 최대 100자
- 분할 우선순위: 문단 → 줄 → 공백 → 문자
- 회사와 PDF 페이지 경계 유지

현재 자료의 최대 입력은 E5 접두사를 포함해 463토큰으로 측정됐다. E5의
512토큰 제한을 넘는 입력은 임베딩 단계에서 오류로 처리한다.

`top_k=3`은 담당자와 합의한 초기값이며 검색 품질상 최적값임을 의미하지
않는다. 평가 질문으로 recall과 불필요 청크 비율을 측정한 뒤 조정해야 한다.

## 10. 오류 처리

다음 문제는 `[]`로 숨기지 않고 예외를 발생시킨다.

- 벡터 DB 또는 manifest 누락
- PDF 변경에 따른 SHA-256 불일치
- 인덱스 스키마·임베딩 모델·거리 설정 불일치
- FAISS 벡터 개수 또는 차원 불일치
- 다른 기업 문서 혼입
- 필수 문서명·경로·페이지 메타데이터 누락
- 임베딩 하위 프로세스 실패
- FAISS 파일 손상 또는 검색 실패

Agent는 초기화와 검색 호출을 모두 오류 기록 범위에 포함해야 한다.

```python
try:
    retriever = ProductTechRAGRetriever(top_k=3)
    documents = retriever.search(
        company_name="해줌",
        queries=["발전량 예측 기술"],
    )
except Exception:
    logger.exception("제품·기술 RAG 검색 실패")
    raise
```

## 11. 검증 결과

- 로컬 인덱스 환경: 테스트 11개 통과
- 인덱스가 없는 새 pull 환경: 테스트 10개 통과
- 생성 인덱스 무결성 검사 1개는 DB 생성 전 정상적으로 건너뜀
- 실제 해줌 검색 결과를 담당자의 `RetrievedDocument` 계약으로 변환
- 담당자의 `adapt_retrieval_result`와 `prepare_documents` 단계 통과
- 검색 원문과 PDF 페이지 일치 확인
- 기업별 인덱스 혼입 방지 확인
- 복수 질의와 동일 청크 중복 제거 확인
- 검색 장애의 예외 전파 확인

검증 명령:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v
```

담당자가 제공한 코드가 일부 잘린 상태였고 공통 `evaluation_support`와
프롬프트 의존성이 제공되지 않아 Agent 전체 실행은 검증하지 못했다. 최종
통합 시 완성된 `tech_brief.py`와 공통 모델을 포함해 한 번 더 실행해야 한다.

## 12. 데이터 한계

현재 코퍼스는 기업이 직접 작성한 원본 제품 설계서가 아니라 공개 자료를
편집한 기술 자료집이다. 다음 항목은 검색 결과만으로 확정할 수 없다.

- 내부 시스템 구조와 실제 구현 코드
- 최신 설계 버전과 변경 이력
- 비공개 성능 수치
- 공개되지 않은 특허명·등록번호
- 독립 시험 결과
- 비공개 계약·납품 실적

자료집에는 `[기술 해석]`과 `[분석]` 문구가 포함된다. 담당 Agent는 이를
기업이 검증한 사실이나 실제 구현 내용으로 취급하지 않아야 한다.

기본 유사도 임계값은 설정하지 않았다. 관련 자료가 없는 질문에도 가장
가까운 청크가 반환될 수 있다. 반환 청크의 실제 관련성과 근거 충분성은
Agent 또는 별도 평가 단계에서 판단해야 한다.
