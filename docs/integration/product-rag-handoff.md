# 제품/설계 요약 Agent — RAG 연동 매뉴얼

작성일: 2026-09-30 · 대상: Agent 통합 담당자 · 계약: 현재 구현 기준

## 0. 담당자 ProductTechAgent 연결 — 권장 진입점

첨부된 담당자 코드는 `retriever.search(company_name=..., queries=[...])`를 호출하고 `RetrievedDocument` 필드를 받는다. 아래 객체를 주입하면 담당자의 기본 `adapt_retrieval_result`를 수정하지 않아도 된다. 이후 절의 `create_rag_search`는 저수준 6필드 인터페이스로 유지한다.

```python
from rag.product_tech_retriever import ProductTechRAGRetriever

retriever = ProductTechRAGRetriever(top_k=3)
documents = retriever.search(
    company_name="해줌",
    queries=["발전량 예측 API의 입력 데이터와 출력 결과", "사용자 이용 절차"],
)
```

담당자의 완성된 Agent 프로젝트에서 다음과 같이 주입한다. 아래 `model`은 Agent 담당자가 준비한 `invoke` 지원 모델이다. 현재 RAG 저장소의 `agents/tech_brief.py`는 원래의 빈 파일로 유지되어 이 Agent 생성 예제는 담당자 프로젝트에서 실행해야 한다.

```python
from agents.tech_brief import ProductTechAgent
from rag.product_tech_retriever import ProductTechRAGRetriever

agent = ProductTechAgent(model=model, retriever=ProductTechRAGRetriever(top_k=3))
```

반환 필드 매핑:

| 담당자 필드 | 제공 값 |
|---|---|
| `document_id` | PDF 상대경로. 담당자의 PDF citation.source 검증 기준과 동일 |
| `chunk_id` | 문서 경로·페이지·청크 원문을 직렬화한 SHA-256. 같은 원문/위치는 질의와 무관하게 동일 |
| `content` | 청크 원문 그대로 |
| `source_title` | 검증된 문서명 |
| `page` | 1-based PDF 페이지 |
| `source_url` | `None`. 로컬 경로를 HTTP URL로 위장하지 않음 |
| `metadata.source_type` | `pdf` |
| `metadata.company_name` | 검증된 회사의 호출 입력 이름 그대로. 담당자 prepare_documents의 문자열 비교와 일치 |
| `metadata.company_id` / `canonical_company_name` | 실제 선택한 DB 코드 / 표준 회사명 |
| `metadata.source` / `published_date` | PDF 경로 / 확인된 발행일 또는 None |

`retrieval_score`, `evidence_id`, 점수·요약은 만들지 않는다. `provenance=independent`를 임의로 부여하지 않는다. 담당자 코드가 독립 검증 자료 부족을 표시하는 것은 현재 자료의 한계에 맞는 동작이다.

- `top_k=3`은 **질문별** 최대 3개다. 전체 반환은 최대 `질문 수 × 3`에서 중복을 뺀 수다.
- 질의 입력 순서와 각 질의의 검색 순서를 유지한다. 서로 다른 질의 결과를 하나의 전역 관련도순으로 재평가하지 않는다.
- 빈 질의 목록 또는 공백만 있는 질의 목록은 `[]`. 중복 질의는 한 번만 검색한다.
- 미지원 회사는 `ValueError`. 하나의 질의라도 검색에 실패하면 전체 호출이 예외를 전파하며 부분 성공을 정상 반환하지 않는다.
- 기업별 검색 객체는 재사용하지만 문서 재구축 후에는 새 ProductTechRAGRetriever 객체를 만든다.
- 회사명은 실제 DB 코드로 검증한다. 지원 별칭은 `rag/product_tech_retriever.py`의 resolve_company를 참조한다.

전달 대상은 `rag/` 코드, 원본 PDF, `tech_data/vector_index_product/`, 의존성 파일과 이 매뉴얼이다. DB 폴더만 전달하면 임베딩·회사 라우팅·필드 매핑이 빠져 바로 연결할 수 없다. 전달용 ZIP에는 키·.env·.venv·모델 캐시를 포함하지 않는다.

검증 범위: 담당자 첨부 코드의 RetrievedDocument, adapt_retrieval_result, prepare_documents를 분리해 실제 검색 결과로 검사했다. 첨부 파일 끝이 잘렸고 evaluation_support/프롬프트 의존성이 없어 Agent 전체 실행은 검증하지 못했다. 이 분리 검사에서는 외부 ContractModel 대신 extra=forbid인 Pydantic BaseModel을 사용했다. 완성본의 추가 검증 규칙이 있다면 최종 통합 검증이 필요하다.

실제 반환 샘플: `docs/examples/product-tech-agent-result.json`.

## 1. 제공 범위

소유 경계: `rag/`는 RAG 담당자, `agents/tech_brief.py`는 제품/설계 요약 Agent 담당자가 구현한다. RAG 모듈은 Agent 구현 파일을 import하지 않는다.

기업별 기술 자료를 검색하고 실제 PDF 추출 원문과 페이지를 반환하는 동기식 검색 인터페이스다. 질문 라우팅, LLM 요약, 사실 판정, evidence_id 생성, 평가·점수 산정은 포함하지 않는다.

- 코퍼스: `에너지 스타트업 기술 자료집`, 단일 PDF 50쪽, 10개 기업 × 5쪽.
- 인덱스: 기업별 FAISS 10개, 총 128개 청크.
- 기업 범위는 초기화 시 고정한다. 질문에 다른 기업명을 넣어도 검색 기업은 바뀌지 않는다.
- 반환값은 최대 3개가 기본값이다. 검색 결과가 질문의 근거임을 보증하지 않는다.
- 이 PDF는 공개 자료를 편집한 기술 자료집이다. 기업이 작성한 원본 설계서가 아니며 `[기술 해석]`·`[분석]` 문구가 포함된다. Agent는 이를 검증된 구현 사실과 구분해야 한다.

## 2. 빠른 연결

모든 셸 명령의 작업 디렉터리는 `skala-rag` 프로젝트 루트다. 현재 로컬 경로는 `/Users/sun/Desktop/RAG/code/skala-rag`다.

```bash
cd /Users/sun/Desktop/RAG/code/skala-rag
.venv/bin/python - <<'PYTHON'
from rag.tech_search import create_rag_search

rag_search = create_rag_search("HZ", top_k=3)
results = rag_search("발전량 예측 API의 입력 데이터와 출력 결과, 연동 방식")
for result in results:
    print(result)
PYTHON
```

모듈의 최상위 함수는 `create_rag_search`다. 여기에서 반환한 함수가 아래 계약을 만족한다. `from rag.tech_search import rag_search`는 지원하지 않는다.

```python
from rag.tech_search import create_rag_search

rag_search = create_rag_search("HZ")
# 반환된 함수의 시그니처: rag_search(query: str) -> list[dict]
results = rag_search("제품의 입력 데이터와 처리 결과")
```

초기화는 질문마다 반복하지 말고 기업별로 한 번 수행한다. 여러 기업을 다루는 Agent는 `company_id -> rag_search` 매핑을 보관한다. API는 동기식이므로 비동기 애플리케이션에서는 요청 처리 스레드를 직접 차단하지 않도록 실행 경계를 둔다. 병렬 처리량은 검증하지 않았다.

## 3. 초기화 설정

```text
create_rag_search(
    company_id: str,
    top_k: int = 3,
    min_similarity=None,
    index_dir=INDEX,
)
```

| 설정 | 의미 |
|---|---|
| `company_id` | 아래 기업 코드. 대소문자 무관. 회사명 문자열 대신 코드를 사용한다. |
| `top_k` | 양의 정수. 중복 제거·필터링 이후 반환할 최대 개수. |
| `min_similarity` | `None` 또는 -1~1 범위의 코사인 유사도 하한. 기본값은 미적용. |
| `index_dir` | `manifest.json`과 기업별 하위 폴더를 포함한 인덱스 루트. 기본값은 프로젝트의 `tech_data/vector_index_product`. |

| 코드 | 기업 | 자료집 PDF 페이지 |
|---|---|---|
| EN | 엔라이튼 | 1–5 |
| HZ | 해줌 | 6–10 |
| SH | 식스티헤르츠 | 11–15 |
| SY | 시너지 | 16–20 |
| EC | 인코어드 | 21–25 |
| VP | 브이피피랩 | 26–30 |
| VG | 브이젠 | 31–35 |
| EX | 에너지엑스 | 36–40 |
| CR | 크로커스 | 41–45 |
| LC | 리셀 | 46–50 |

`index_dir`는 다른 PDF를 자동 등록하는 옵션이 아니다. 현재 초기화는 `tech_ingest.PDF`에 지정된 단일 PDF의 SHA-256과 manifest를 비교한다. 다른 문서를 추가하려면 인덱싱 구성부터 확장해야 한다.

## 4. 반환 계약

항목은 아래 여섯 필드만 가진다. LangChain `Document`나 점수는 외부로 반환하지 않는다.

| 필드 | 타입 | 규칙 |
|---|---|---|
| `source_title` | `str` | 인덱싱 시 확인·저장한 실제 문서명. 없으면 오류. |
| `source_type` | `str` | 항상 `pdf`. |
| `source` | `str` | 현재는 프로젝트 루트 기준 PDF 상대경로. 없으면 오류. |
| `excerpt` | `str` | 해당 청크의 `page_content` 그대로. 요약·문장 결합·표현 수정 없음. |
| `page` | `int` | PDF 뷰어 기준 1-based 페이지. 인쇄된 장·절 페이지 번호와는 다를 수 있다. |
| `published_date` | `str` 또는 `None` | 확인된 날짜만 YYYY-MM-DD. 현재 코퍼스는 모두 `None`. |

줄바꿈, 표 추출 순서, 청크 경계의 미완성 문장은 보존된다. `excerpt`는 PDF에서 추출된 텍스트의 원문이며 PDF의 시각적 레이아웃을 재현하지 않는다. 자료집 안의 `[HZ2]` 등은 원문에 있는 참고문헌 표식이다. 이 인터페이스가 생성하는 evidence_id가 아니다.

### 페이지 저장·변환

- 내부 `page`: PyMuPDFLoader의 0-based 값. `page_index_base=0`으로 저장한다.
- 내부 `page_number`: PDF 기준 1-based 값. 현재 어댑터가 우선 사용한다.
- 외부 `page`: `page_number`를 그대로 반환한다. 다시 +1 하지 않는다.
- `page_number`가 없는 문서는 정수 `page`와 명시적인 `page_index_base`가 필요하다. base=0일 때만 +1 한다.
- 필수 페이지·문서명·경로를 반환 시 추정하지 않는다. 초기화 중 전체 문서 메타데이터를 검사한다.
- `published_date`가 있으면 유효한 ISO 날짜인지 검사한다. 문서 기준일·파일명 날짜·버전에서 날짜를 만들어내지 않는다.

어댑터를 개별 사용해야 한다면 다음 함수를 import한다.

```python
from rag.tech_search import document_to_result

result = document_to_result(document)
```

이 어댑터는 메타데이터 변환만 수행한다. 외부 문서의 메타데이터가 실제 PDF와 일치하는지는 그 문서의 인덱싱 담당자가 보장해야 한다.

## 5. 실제 검색 결과 1건

초기화: `create_rag_search("HZ")`

검색어: `발전량 예측 API의 입력 데이터와 출력 결과, 연동 방식`
아래는 실제 반환된 첫 번째 항목이다. 전체 결과 3건의 원문이 해당 PDF 페이지 안에 존재하는지 확인했다.

```json
{
  "source_title": "에너지 스타트업 기술 자료집",
  "source_type": "pdf",
  "source": "tech_data/tech-10-startups-50pages-2026-09-30.pdf",
  "excerpt": "에너지 스타트업 기술 자료집   7\n해줌 Haezoom 2 작동 원리와 기술 구성\nHZ-P2      기술 분석 자료  |  기준일 2026 09 30\n기술 항목\n공개 내용과 적용 조건\n예측 입력\n공개 API에 pid, start, end, ref, timezone, interval, tilt, azimuth 항목. 샘플의 과거 날짜\n를 현행 API 버전으로 추정하지 않음. [HZ2]\n기상 결합\n1~72시간 예측 및 위성·기상·설비 데이터 활용 설명. 학습 데이터 범위·모델 구조·모델별 성\n능 미확보. [HZ2, HZ3]\nDR 계측\n20초 단위 전력량 계량·감축 이행 확인 및 피크 사전 알림을 제품 기능으로 제시. [HZ7]\n전력중개 연계\n계약/서류→검사→필요 시 계량기 전환→시장가입/승인→자원 구성→예측 테스트→운영/정\n산. [HZ6]\n차별성 비교\n예측 API와 발전소 운영·소비자 DR까지 포괄. 엔라이튼·VPPLab 대비 실제 오차 우위는 동일 \n데이터로 미검증.\n시간 정보의 역할  [기술 해석] start와 end는 조회 구간, ref는 예측 기준 시점, timezone은 시각 해석에 관\n련된다. 동일 발전 시간에도 발행 시점이 다른 예측이 여러 개 존재할 수 있다. date 배열과 예측값 배열의 대\n응을 유지하고 발전소 ID와 기준 시점을 함께 보관해야 하루 전 예측과 직전 예측을 분리할 수 있다. 공개 샘\n플을 바탕으로 한 연동 해석이며 내부 저장 구조를 뜻하지 않는다. [HZ2]\n설비 조건과 예측  [기술 해석] 같은 지역이라도 패널의 기울기·방위각이 다르면 받는 일사량이 달라진다. 기",
  "page": 7,
  "published_date": null
}
```

동일 예시 파일: `docs/examples/product-rag-result.json`.

## 6. 무결과와 장애 처리

| 상황 | 동작 |
|---|---|
| 빈 문자열·공백 질문 | `[]` |
| 후보 없음 또는 모든 후보가 설정한 유사도 하한 미달 | `[]` |
| 질문이 문자열이 아님 | `TypeError` |
| 잘못된 기업 코드·top_k·유사도 범위 | `ValueError` |
| manifest 또는 PDF 파일 없음 | `FileNotFoundError` 등 파일 예외 |
| manifest 파싱 실패 | JSON 파싱 예외 |
| 스키마·모델·거리 설정·PDF 해시·벡터 개수·차원 불일치 | `ValueError` |
| 필수 메타데이터 누락·다른 기업 문서 혼입 | `ValueError` |
| FAISS 파일 손상·로딩 실패 | FAISS/파일 계층 예외를 그대로 전파 |
| 임베딩 하위 프로세스 실패 | `RuntimeError` |

```python
from rag.tech_search import create_rag_search

rag_search = create_rag_search("HZ")
assert rag_search("   ") == []

# 존재하지 않는 경로: []가 아니라 FileNotFoundError가 발생한다.
create_rag_search("HZ", index_dir="/nonexistent/product-rag-index")
```

Agent는 초기화와 검색 호출을 모두 오류 기록 범위에 포함해야 한다. 운영 로깅에서 예외를 기록하더라도 빈 결과로 바꾸지 않는다.

```python
import logging
from rag.tech_search import create_rag_search

logger = logging.getLogger(__name__)
try:
    rag_search = create_rag_search("HZ")
    evidence = rag_search("사용자 이용 절차와 API 연동 조건")
except Exception:
    logger.exception("제품 RAG 실패: company_id=HZ")
    raise
```

기본 `min_similarity=None`에서는 관련 자료가 없어도 가장 가까운 후보가 나올 수 있다. `[]`를 의미적 근거 부족의 완전한 판정으로 사용하면 안 된다. `리셀 특허`처럼 코퍼스에 특허 정보가 없는 질문은 반환된 기술 설명을 특허 근거로 간주하지 않는다.

## 7. 검색·청크 기준

- 임베딩: `intfloat/multilingual-e5-small`, 384차원, 정규화 적용.
- 문서 접두사: `passage: `, 질문 접두사: `query: `.
- 분할: 최대 800자, 중첩 최대 100자. 회사·PDF 페이지 경계를 넘지 않는다.
- 문단 → 줄 → 공백 → 문자 순으로 분할한다. 현재 자료의 최대 입력은 접두사 포함 463토큰으로 측정됐고, 실행 시 512토큰 초과 입력은 오류 처리한다.
- FAISS는 정규화 벡터의 squared L2 거리를 사용한다. 내부 코사인 유사도는 `1 - distance / 2`로 계산한다.
- 기업 인덱스 전체를 후보로 조회한 뒤 순위 보정·유사도 필터·중복 제거를 거쳐 상위 `top_k`를 반환한다.
- 중복 기준은 `(source, page, excerpt)`의 정확한 일치다. 겹치는 청크나 의미상 중복까지 제거하지 않는다.

본문 우선 보정은 `content_kind`가 `references`, `heading`, `toc`, `market`인 후보의 거리에 0.04를 더한다. 이는 작은 관련도 차이에서 본문을 우선하기 위한 휴리스틱이며 평가로 최적화된 값은 아니다. 현재 인덱싱은 페이지의 `근거 원문` 경계 이후 청크를 `references`, 나머지를 `body`로 분류한다. 제목·목차·시장 자료의 범용 자동 분류기는 구현하지 않았다.

기본 상위 3개는 Agent 연동 계약에 맞춘 값이다. 검색 품질상 최적임을 의미하지 않는다. 평가셋으로 top_k와 임계값을 결정해야 한다.

## 8. 런타임과 전달 파일

검증 환경: Python 3.11.16, 프로젝트 `.venv`. 필수 API 키는 없다.

새 환경에서는 같은 Python 버전으로 가상환경을 만들고 잠금 파일을 설치한다.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-tech.lock.txt
```

필요 파일:

```text
rag/__init__.py
rag/product_tech_retriever.py
rag/tech_ingest.py
rag/tech_search.py
rag/tech_embeddings.py
requirements-tech.txt
requirements-tech.lock.txt
tech_data/tech-10-startups-50pages-2026-09-30.pdf
tech_data/vector_index_product/
  manifest.json
  EN/index.faiss
  EN/index.pkl
  ... 나머지 9개 기업의 index.faiss 및 index.pkl
```

- 기본 모델 캐시: 프로젝트 `.cache/huggingface`. `HF_HOME`이 이미 있으면 그 설정을 따른다.
- `HF_HUB_OFFLINE=1`: 모델이 완전히 캐시된 경우에만 사용한다. 새 환경에서는 최초 모델 다운로드가 필요하다.
- 기존 `.venv`를 다른 PC에 복사하지 말고 재생성한다.
- 벡터 인덱스와 모델 캐시는 Git ignore 대상이므로 저장소만 복제하면 없을 수 있다. 신뢰할 수 있는 인덱스를 별도로 전달하거나 재구축한다.
- FAISS의 `.pkl`은 역직렬화를 수반한다. 이 프로젝트에서 직접 생성했거나 신뢰할 수 있는 경로로 전달받은 파일만 사용한다.
- Mac의 FAISS/PyTorch OpenMP 충돌을 피하기 위해 E5는 별도 프로세스에서 실행한다. 현재는 질의마다 모델 프로세스를 시작하므로 고처리량 서비스용 지연 최적화는 되어 있지 않다.

## 9. 재구축·검증

DB가 없는 새 환경에서:

```bash
.venv/bin/python rag/tech_ingest.py
```

이미 같은 출력 경로가 있으면 `FileExistsError`로 중단한다. 기존 인덱스를 덮어쓰지 않는다. 재구축하려면 새 출력 경로를 사용한다.

```python
from pathlib import Path
from rag.tech_ingest import build_vector_db
from rag.tech_search import create_rag_search

new_index = Path("tech_data/vector_index_product_next")
build_vector_db(output_dir=new_index)
rag_search = create_rag_search("HZ", index_dir=new_index)
```

PDF 또는 메타데이터를 바꿨으면 재구축하고 검색 함수를 다시 초기화한다. 현재 PDF 구성은 회사 순서와 기업당 5쪽을 전제로 검증한다. 임의의 PDF를 교체해 바로 사용하도록 일반화되어 있지 않다.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v
```

기본 인덱스 경로 기준 테스트 11개 통과 및 실제 해줌 검색 원문·페이지 대조를 확인했다. 생성 인덱스가 없는 환경에서는 저장 벡터 무결성 테스트 한 건을 건너뛴다. 테스트는 빈 결과·임계값·중복·오류 전파·본문 우선 보정·메타데이터·페이지 변환·저장 벡터 무결성을 포함한다. 코드 테스트 통과는 의미적 검색 정확도나 사실성 평가의 통과를 뜻하지 않는다.

## 10. 미지원 범위와 Agent 책임

- 다중 원본 설계서와 버전 계보, 최신 버전 선택, 현재 반영 여부, 충돌하는 설계 버전의 분리 반환은 미구현이다. 현재 단일 PDF의 `document_version`은 `None`이며 `manifest.schema_version=2`는 인덱스 메타데이터 스키마 번호다.
- 서로 다른 설계 버전을 연결하려면 인덱싱에 문서별 식별자·버전·현재성 근거를 먼저 추가해야 한다. 날짜나 파일명만으로 현재 설계를 추정하지 않는다.
- 시장 문서 통합 검색이나 최근 3년 필터는 없다. 현재 검색은 선택 기업의 기술 자료집 범위다.
- 비기능 요구사항·보안·기술 스택·PoC 변경 이력 등 자료집에 없는 항목은 검색 설정으로 보충할 수 없다.
- source의 PDF 페이지는 편집 자료집의 페이지다. 자료집이 인용한 외부 웹 문서 또는 원본 회사 설계서의 페이지가 아니다.
- Agent는 evidence_id, 원문에 근거한 요약, 출처 인용, 근거 부족·충돌 판단을 담당한다.
- 질문 의도와 반환된 원문의 실제 관련성, 기술 해석과 회사 실적의 구분은 Agent 및 평가셋에서 검증해야 한다.
