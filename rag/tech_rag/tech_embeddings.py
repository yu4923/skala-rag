"""로컬 E5를 LangChain에 연결한다. Mac의 OpenMP 충돌 때문에 별도 실행한다."""
import json
import os
from pathlib import Path
import subprocess
import sys

from langchain_core.embeddings import Embeddings


class LocalE5Embeddings(Embeddings):
    """강의의 OpenAIEmbeddings와 같은 인터페이스로 로컬 벡터를 제공한다."""

    def embed_documents(self, texts):
        """문서에 E5 검색용 접두사를 붙여 벡터 목록으로 변환한다."""
        return self._encode(["passage: " + text for text in texts])

    def embed_query(self, text):
        """질문에 E5 질의용 접두사를 붙여 벡터 하나로 변환한다."""
        return self._encode(["query: " + text])[0]

    def _encode(self, texts):
        """FAISS 프로세스에 PyTorch를 불러오지 않고 임베딩만 요청한다."""
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            input=json.dumps(texts), capture_output=True, text=True,
        )
        if process.returncode:
            raise RuntimeError(f"로컬 임베딩 실패:\n{process.stderr}")
        return json.loads(process.stdout)


def encode_in_worker():
    """분리된 프로세스에서 토큰 한도를 확인하고 E5 임베딩을 계산한다."""
    repository_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("HF_HOME", str(repository_root / ".cache/huggingface"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(1)
    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    texts = json.load(sys.stdin)
    for text in texts:
        if len(model.tokenizer(text)["input_ids"]) > 512:
            raise ValueError("E5는 512토큰까지 처리합니다. 청크나 질문을 줄이세요.")
    vectors = model.encode(texts, normalize_embeddings=True, batch_size=16)
    print(json.dumps(vectors.tolist()))


if __name__ == "__main__":
    encode_in_worker()
