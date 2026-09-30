"""PDF → 청크 → E5 임베딩 → 기업별 FAISS 저장까지만 실행한다."""
import hashlib
import json
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __package__:
    from .tech_embeddings import LocalE5Embeddings
else:
    from tech_embeddings import LocalE5Embeddings

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "rag/data/tech/tech-10-startups-50pages-2026-09-30.pdf"
INDEX = ROOT / "rag/indexes/product"
SOURCE_TITLE = "에너지 스타트업 기술 자료집"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
COMPANIES = {
    "EN": "엔라이튼", "HZ": "해줌", "SH": "식스티헤르츠", "SY": "시너지",
    "EC": "인코어드", "VP": "브이피피랩", "VG": "브이젠", "EX": "에너지엑스",
    "CR": "크로커스", "LC": "리셀",
}

def load_documents():
    """PDF 50쪽을 Document 목록으로 읽고 회사와 페이지 정보를 붙인다."""
    loader = PyMuPDFLoader(str(PDF))
    docs = loader.load()
    if len(docs) != 50:
        raise ValueError("기업당 5쪽, 총 50쪽인 자료집이 필요합니다.")

    for i, doc in enumerate(docs):
        company_id = list(COMPANIES)[i // 5]
        marker = f"{company_id}-P{i % 5 + 1}"
        if marker not in doc.page_content:
            raise ValueError(f"{i + 1}쪽의 기업 표식이 {marker}와 다릅니다.")
        if SOURCE_TITLE not in doc.page_content:
            raise ValueError("PDF 본문에서 등록한 문서명을 확인할 수 없습니다.")
        doc.metadata.update(
            source_title=SOURCE_TITLE,
            source=str(PDF.relative_to(ROOT)),
            page_index_base=0,
            published_date=None,
            document_version=None,
            company_id=company_id,
            company_name=COMPANIES[company_id],
            page_number=i + 1,                                          # PDF 뷰어와 같은 1부터 시작하는 쪽 번호다.
            page_context=doc.page_content,                              # 잘린 문장의 전후 문맥을 확인할 때 쓴다.
        )
    return docs

def split_documents(docs):
    """페이지 경계를 지키고 문단·줄·단어 순서로 청크를 나눈다."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = f"{doc.metadata['company_id']}-{i + 1}"
        reference_start = doc.metadata["page_context"].find("근거 원문")
        # URL 목록만 있는 청크는 기술 설명 본문보다 낮은 우선순위를 준다.
        doc.metadata["content_kind"] = (
            "references" if reference_start >= 0
            and doc.metadata["start_index"] >= reference_start else "body"
        )
    return chunks


def build_vector_db(output_dir=INDEX):
    """10개 기업의 문서를 임베딩하고 각 회사의 FAISS 파일을 저장한다."""
    if output_dir.exists():
        raise FileExistsError("기존 DB를 보호합니다. 새로 구축하려면 INDEX 경로를 바꾸세요.")
    docs = load_documents()
    print(f"[1/3] PDF {len(docs)}쪽 로딩", flush=True)
    chunks = split_documents(docs)
    print(f"[2/3] {len(chunks)}개 청크 생성 (최대 {CHUNK_SIZE}자)", flush=True)
    embeddings = LocalE5Embeddings()
    # E5 모델은 로딩 비용이 크므로 전체 청크를 한 번에 임베딩한다.
    vectors = embeddings.embed_documents([doc.page_content for doc in chunks])
    counts = {}
    for code, name in COMPANIES.items():
        company_items = [
            (doc, vector)
            for doc, vector in zip(chunks, vectors)
            if doc.metadata["company_id"] == code
        ]
        company_chunks = [doc for doc, _vector in company_items]
        vectorstore = FAISS.from_embeddings(
            [(doc.page_content, vector) for doc, vector in company_items],
            embeddings,
            metadatas=[doc.metadata for doc in company_chunks],
        )
        vectorstore.save_local(str(output_dir / code))
        counts[code] = len(company_chunks)
        print(f"[3/3] {name}: {len(company_chunks)}개 벡터 저장", flush=True)

    # 완료 기록은 마지막에 저장해 중단된 구축과 구분한다.
    manifest = dict(
        schema_version=2,
        source=PDF.name, source_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
        model="intfloat/multilingual-e5-small", dimension=384,
        normalized=True, distance="squared_l2",
        chunk_size_chars=CHUNK_SIZE, chunk_overlap_chars=CHUNK_OVERLAP,
        pages=len(docs), chunks=len(chunks), companies=counts,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"완료: {output_dir}")
    return manifest


if __name__ == "__main__":
    build_vector_db()
