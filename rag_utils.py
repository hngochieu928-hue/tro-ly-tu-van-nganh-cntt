# ============================================================
# RAG_UTILS.PY — HYBRID + RERANK + MULTI-LLM
# Tối ưu: Chính xác + Nhanh + Chi tiết
# ============================================================

import re
import unicodedata
from functools import lru_cache

import chromadb
import ollama

from external_llm import stream_external
from source_info import get_source_info, make_snippet

from config import (
    DB_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    COLLECTION_NAME,
    MAX_CONTEXT_CHARS,
    MAX_DISTANCE,
    EMBEDDING_CACHE_SIZE,
    OLLAMA_KEEP_ALIVE,
    LLM_TEMPERATURE,
    LLM_NUM_PREDICT,
    LLM_NUM_CTX,
    OLLAMA_NUM_THREAD,
    USE_HYBRID,
    USE_RERANKER,
    CANDIDATES_K,
    FINAL_TOP_K,
    RRF_K,
    RERANKER_MODEL,
    RERANKER_MAX_LENGTH,
)


# ============================================================
# 1. KẾT NỐI CHROMADB
# ============================================================

@lru_cache(maxsize=1)
def get_chroma_client():
    try:
        return chromadb.PersistentClient(path=DB_DIR)
    except Exception as e:
        raise RuntimeError(f"Không thể kết nối ChromaDB: {e}")


@lru_cache(maxsize=1)
def get_chroma_collection():
    try:
        client = get_chroma_client()
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Không thể mở collection '{COLLECTION_NAME}': {e}"
        )


# ============================================================
# 2. CHUẨN HÓA CÂU HỎI
# ============================================================

def normalize_question(question):
    if not question:
        return ""
    return re.sub(r"\s+", " ", question.strip())


# ============================================================
# 2b. TỰ ĐỘNG PHÁT HIỆN BỘ LỌC METADATA (AUTO FILTER)
# ============================================================

_CATEGORY_RULES = [
    (["ky tuc xa", "ktx"], "ktx"),
    (["hoc bong"], "hoc_bong"),
    (["mien giam hoc phi", "ho tro chi phi hoc tap"], "mien_giam_hoc_phi"),
    (["ky luat"], "ky_luat"),
    (["dong phuc", "le phuc"], "dong_phuc"),
    (["ren luyen"], "ren_luyen"),
    (["quy doi diem", "hoc ba sang thpt", "quy doi hoc ba"], "quy_doi_diem"),
]


def _to_chroma_where(where):
    if not where:
        return None
    if len(where) == 1:
        key, value = next(iter(where.items()))
        return {key: value}
    return {"$and": [{k: v} for k, v in where.items()]}


def detect_doc_filter(question):
    """Đoán bộ lọc doc_type/category/year từ câu hỏi để tìm kiếm chính xác
    hơn khi kho dữ liệu có nhiều văn bản chồng chéo chủ đề. Trả về None
    nếu không chắc chắn — khi đó tìm kiếm không lọc như bình thường."""
    q = _strip_diacritics(normalize_question(question) or "")
    if not q:
        return None

    if "diem chuan" in q:
        where = {"doc_type": "diem_chuan"}
        year_match = re.search(r"20(2[4-9]|3[0-9])", q)
        if year_match:
            where["year"] = year_match.group()
        return where

    if "ho so" in q and "nhap hoc" in q:
        return {"doc_type": "tuyen_sinh"}

    for keywords, category in _CATEGORY_RULES:
        if any(kw in q for kw in keywords):
            return {"category": category}

    return None


# ============================================================
# 2c. QUY ĐỔI ĐIỂM HỌC BẠ → THPT (tính toán chính xác, không để LLM đoán)
# ============================================================

def detect_score_conversion(question):
    q = _strip_diacritics(normalize_question(question) or "")
    if "quy doi" not in q or "hoc ba" not in q:
        return None

    numbers = re.findall(r"\d{1,2}(?:[.,]\d{1,2})?", normalize_question(question))
    if not numbers:
        return None

    try:
        x = float(numbers[0].replace(",", "."))
    except ValueError:
        return None

    try:
        from score_converter import convert_with_priority, DEFAULT_PERCENTILE_BANDS
    except ImportError:
        return None

    m, n = DEFAULT_PERCENTILE_BANDS["hoc_ba"]
    a, b = DEFAULT_PERCENTILE_BANDS["thpt"]
    if not (m <= x <= n):
        return None

    try:
        result = convert_with_priority(x)
    except Exception:
        return None

    return (
        "[KẾT QUẢ TÍNH TOÁN THEO CÔNG THỨC QUY ĐỔI ĐIỂM CHÍNH THỨC CỦA TRƯỜNG]\n"
        f"Điểm học bạ x = {result['x_hoc_ba']} → điểm THPT tương đương y = "
        f"{result['y_thpt_tuong_duong']} (công thức y = a + (x-m)/(n-m)×(b-a), "
        f"với mốc VÍ DỤ MINH HỌA [m,n]=[{m},{n}] và [a,b]=[{a},{b}]).\n"
        "LƯU Ý BẮT BUỘC PHẢI NÊU RÕ: đây là mốc phân vị ví dụ minh họa trong quy "
        "định, Nhà trường CHƯA công bố mốc phân vị chính thức năm 2026 (sẽ công "
        "bố sau khi có kết quả và phổ điểm thi tốt nghiệp THPT năm 2026). Không "
        "được trình bày con số này như kết quả quy đổi chính thức cuối cùng."
    )


# ============================================================
# 3. TOKENIZE TIẾNG VIỆT CHO BM25
# ============================================================

def _strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text if unicodedata.category(c) != "Mn"
    )
    text = text.replace("đ", "d").replace("Đ", "D")
    return text.lower()


def _tokenize_vn(text: str):
    if not text:
        return []

    cleaned = re.sub(r"[^\w\s]", " ", text.lower())

    tokens_with = cleaned.split()
    tokens_without = _strip_diacritics(cleaned).split()

    seen, result = set(), []
    for tok in tokens_with + tokens_without:
        if tok and tok not in seen:
            seen.add(tok)
            result.append(tok)
    return result


# ============================================================
# 4. BM25 INDEX
# ============================================================

_bm25_index = None
_bm25_corpus = None
_bm25_metadatas = None


def _build_bm25_index(collection):
    global _bm25_index, _bm25_corpus, _bm25_metadatas

    if _bm25_index is not None:
        return

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise RuntimeError(
            "Chưa cài rank_bm25. Chạy: pip install rank_bm25"
        )

    data = collection.get(include=["documents", "metadatas"])
    _bm25_corpus = data.get("documents", []) or []
    _bm25_metadatas = data.get("metadatas", []) or []

    if not _bm25_corpus:
        raise RuntimeError("Collection rỗng — chưa build DB?")

    tokenized = [_tokenize_vn(doc) for doc in _bm25_corpus]
    _bm25_index = BM25Okapi(tokenized)


def _matches_where(meta, where):
    if not where:
        return True
    return all(meta.get(k) == v for k, v in where.items())


def bm25_search(collection, question, top_k=CANDIDATES_K, where=None):
    _build_bm25_index(collection)

    if _bm25_index is None:
        return [], []

    query_tokens = _tokenize_vn(question)
    if not query_tokens:
        return [], []

    scores = _bm25_index.get_scores(query_tokens)

    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)

    documents, metadatas = [], []
    for i, _ in indexed:
        meta = _bm25_metadatas[i]
        if not _matches_where(meta, where):
            continue
        documents.append(_bm25_corpus[i])
        metadatas.append(meta)
        if len(documents) >= top_k:
            break

    return documents, metadatas


# ============================================================
# 5. RERANKER
# ============================================================

_reranker = None


def get_reranker():
    global _reranker

    if _reranker is not None:
        return _reranker

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise RuntimeError(
            "Chưa cài sentence-transformers. "
            "Chạy: pip install sentence-transformers"
        )

    _reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=RERANKER_MAX_LENGTH,
    )
    return _reranker


def rerank(question, documents, metadatas, top_k=FINAL_TOP_K):
    if not documents:
        return [], [], []

    try:
        model = get_reranker()
    except Exception:
        return (
            documents[:top_k],
            metadatas[:top_k],
            [0.0] * min(top_k, len(documents)),
        )

    pairs = [(question, doc) for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(
        zip(scores, documents, metadatas),
        key=lambda x: float(x[0]),
        reverse=True,
    )[:top_k]

    return (
        [d for _, d, _ in ranked],
        [m for _, _, m in ranked],
        [float(s) for s, _, _ in ranked],
    )


# ============================================================
# 6. EMBEDDING
# ============================================================

_embedder = None


def get_embedder():
    """Model embedding chạy local qua sentence-transformers (không cần Ollama)."""
    global _embedder

    if _embedder is not None:
        return _embedder

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "Chưa cài sentence-transformers. "
            "Chạy: pip install sentence-transformers"
        )

    _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


@lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
def create_embedding(question):
    question = normalize_question(question)
    if not question:
        raise ValueError("Câu hỏi không được để trống.")

    try:
        model = get_embedder()
        vector = model.encode(question, normalize_embeddings=True)
    except Exception as e:
        raise RuntimeError(f"Lỗi tạo embedding: {e}")

    return vector.tolist()


# ============================================================
# 7. VECTOR SEARCH
# ============================================================

def vector_search(collection, question, top_k=CANDIDATES_K, where=None):
    question = normalize_question(question)
    if not question:
        return [], [], []

    query_embedding = create_embedding(question)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=_to_chroma_where(where),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise RuntimeError(f"Lỗi truy vấn ChromaDB: {e}")

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    f_docs, f_metas, f_dists = [], [], []
    for doc, meta, dist in zip(documents, metadatas, distances):
        if dist is None:
            continue
        if dist <= MAX_DISTANCE:
            f_docs.append(doc)
            f_metas.append(meta)
            f_dists.append(dist)

    return f_docs, f_metas, f_dists


# ============================================================
# 8. HYBRID SEARCH
# ============================================================

def _rrf_merge(vec_docs, vec_metas, bm25_docs, bm25_metas, k=RRF_K):
    score_map, doc_map, meta_map = {}, {}, {}

    def add_ranked(docs, metas):
        for rank, (doc, meta) in enumerate(zip(docs, metas)):
            key = (meta.get("source", ""), meta.get("chunk", ""))
            score_map[key] = (
                score_map.get(key, 0.0) + 1.0 / (k + rank + 1)
            )
            doc_map[key] = doc
            meta_map[key] = meta

    add_ranked(vec_docs, vec_metas)
    add_ranked(bm25_docs, bm25_metas)

    sorted_keys = sorted(
        score_map.keys(),
        key=lambda kk: score_map[kk],
        reverse=True,
    )

    return (
        [doc_map[kk] for kk in sorted_keys],
        [meta_map[kk] for kk in sorted_keys],
    )


def hybrid_search_documents(collection, question, top_k=FINAL_TOP_K, where=None):
    question = normalize_question(question)
    if not question:
        return [], [], []

    try:
        vec_docs, vec_metas, _ = vector_search(
            collection, question, top_k=CANDIDATES_K, where=where
        )
    except Exception:
        vec_docs, vec_metas = [], []

    bm25_docs, bm25_metas = [], []
    if USE_HYBRID:
        try:
            bm25_docs, bm25_metas = bm25_search(
                collection, question, top_k=CANDIDATES_K, where=where
            )
        except Exception:
            pass

    if USE_HYBRID and bm25_docs:
        merged_docs, merged_metas = _rrf_merge(
            vec_docs, vec_metas, bm25_docs, bm25_metas
        )
    else:
        merged_docs, merged_metas = vec_docs, vec_metas

    if not merged_docs:
        return [], [], []

    if USE_RERANKER:
        docs, metas, scores = rerank(
            question, merged_docs, merged_metas, top_k=top_k
        )
    else:
        docs = merged_docs[:top_k]
        metas = merged_metas[:top_k]
        scores = [0.0] * len(docs)

    for meta, score in zip(metas, scores):
        meta["score"] = float(score)

    return docs, metas, scores


def search_documents(collection, question, top_k=FINAL_TOP_K):
    return hybrid_search_documents(collection, question, top_k=top_k)


# ============================================================
# 9. CẮT THÔNG MINH THEO RANH GIỚI CÂU
# ============================================================

def _smart_truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text

    truncated = text[:max_len]
    last_punct = max(
        truncated.rfind(". "),
        truncated.rfind(".\n"),
        truncated.rfind("!\n"),
        truncated.rfind("?\n"),
        truncated.rfind("\n\n"),
    )

    if last_punct > max_len * 0.6:
        return text[:last_punct + 1].rstrip() + "…"

    last_space = truncated.rfind(" ")
    if last_space > 0:
        return text[:last_space].rstrip() + "…"

    return truncated + "…"


# ============================================================
# 10. TẠO CONTEXT — CHI TIẾT, SẮP THEO SCORE
# ============================================================

def build_context_ex(documents, metadatas, max_chars=MAX_CONTEXT_CHARS):
    """
    Xây context và trả thêm danh sách chỉ số (trong `documents`) của các
    đoạn thực sự được đưa vào context, theo đúng thứ tự [TÀI LIỆU 1..N]
    để giao diện đánh số nguồn khớp với số LLM trích dẫn.
    """
    if not documents:
        return "", []

    indexed = list(enumerate(zip(documents, metadatas)))
    indexed.sort(
        key=lambda x: x[1][1].get("score", 0.0),
        reverse=True,
    )

    context_parts = []
    used = []
    total_chars = 0

    for orig_idx, (document, metadata) in indexed:
        if not document or total_chars >= max_chars:
            break

        info = get_source_info(metadata.get("source", ""))
        ref = f" ({info['ref']})" if info.get("ref") else ""

        header = (
            f"[TÀI LIỆU {len(context_parts) + 1}] "
            f"Nguồn: {info['title']}{ref}\n"
        )

        body = document.strip()
        remaining = max_chars - total_chars - len(header) - 4

        if len(body) > remaining:
            body = _smart_truncate(body, remaining)

        part = f"{header}{body}\n\n"
        context_parts.append(part)
        used.append(orig_idx)
        total_chars += len(part)

    return "".join(context_parts), used


def build_context(documents, metadatas, max_chars=MAX_CONTEXT_CHARS):
    return build_context_ex(documents, metadatas, max_chars)[0]


# ============================================================
# 11. SYSTEM PROMPT — YÊU CẦU CHI TIẾT + CHÍNH XÁC
# ============================================================

SYSTEM_PROMPT = """Bạn là TRỢ LÝ TƯ VẤN NGÀNH VÀ CHUYÊN NGÀNH của Khoa Công nghệ thông tin.

⚠️ QUY TẮC NGÔN NGỮ BẮT BUỘC:
- LUÔN LUÔN trả lời bằng TIẾNG VIỆT 100%.
- TUYỆT ĐỐI KHÔNG dùng tiếng Indonesia, tiếng Trung.
- Được phép dùng thuật ngữ kỹ thuật tiếng Anh phổ biến: AI, IoT, ERP, Cloud, LAN, WAN, Back-End, Front-End, Mobile, Web, Blockchain, Big Data, Machine Learning, Deep Learning, DevOps, Database, API, UI/UX.

📚 CẤU TRÚC ĐÀO TẠO (ghi nhớ chính xác):
Khoa có 2 NGÀNH:
1. NGÀNH CÔNG NGHỆ THÔNG TIN — mã ngành 7480201
   Gồm 3 CHUYÊN NGÀNH:
   - Công nghệ phần mềm
   - Hệ thống thương mại điện tử
   - An ninh mạng
2. NGÀNH KHOA HỌC DỮ LIỆU — là NGÀNH RIÊNG, KHÔNG thuộc CNTT.

⚠️ KHÔNG được nhầm NGÀNH với CHUYÊN NGÀNH.

🎯 NGUYÊN TẮC TRẢ LỜI:
1. Chỉ trả lời dựa trên CONTEXT được cung cấp. KHÔNG tự bịa.
2. Nếu CONTEXT không đủ → nói rõ "chưa tìm thấy thông tin phù hợp".
3. **In đậm** thông tin quan trọng.
4. Sau mỗi ý/số liệu lấy từ CONTEXT, ghi nguồn ngay cuối câu đúng dạng (theo tài liệu N) với N là số trong [TÀI LIỆU N]; nhiều nguồn thì viết (theo tài liệu 2, 5). Chỉ dùng số N có trong CONTEXT, không tự bịa số.

📝 ĐỊNH DẠNG TRẢ LỜI (bắt buộc):
- **Mở đầu**: 1-2 câu tóm tắt câu trả lời.
- **Thân bài**: Trình bày CHI TIẾT, có cấu trúc:
  * Dùng heading `###` cho từng mục
  * Dùng bullet list cho danh sách
  * Dùng **bold** cho từ khóa
  * Dùng bảng khi so sánh
- **Kết luận**: 1-2 câu tóm tắt hoặc gợi ý.
- Độ dài: **300-500 từ** cho câu hỏi phức tạp, **150-300 từ** cho câu hỏi đơn giản.

📋 VÍ DỤ ĐỊNH DẠNG ĐÚNG:

### Chuyên ngành phù hợp
Nếu bạn **thích lập trình và xây dựng phần mềm**, chuyên ngành **Công nghệ phần mềm** là lựa chọn hàng đầu.

### Lý do phù hợp
- **Đào tạo chuyên sâu**: tập trung vào lập trình, thuật toán, kiến trúc phần mềm.
- **Đa nền tảng**: Desktop, Web, Mobile.
- **Nghề nghiệp đa dạng**: Back-End, Front-End, Full-Stack, Tester.

### Gợi ý thêm
Nếu bạn cũng quan tâm đến **AI và dữ liệu**, có thể cân nhắc ngành **Khoa học dữ liệu**.

---
"""


# ============================================================
# 12. TẠO PROMPT
# ============================================================

def build_prompt(question, context, history=None):
    prompt = SYSTEM_PROMPT
    prompt += "\n\n========== CONTEXT (dùng để trả lời) ==========\n\n"
    prompt += context

    if history:
        prompt += "\n\n========== LỊCH SỬ HỘI THOẠI ==========\n\n"
        for message in history[-6:]:
            role = message.get("role", "user")
            content = message.get("content", "")[:400]
            speaker = "Sinh viên" if role == "user" else "Trợ lý"
            prompt += f"{speaker}: {content}\n"

    prompt += "\n\n========== CÂU HỎI CỦA SINH VIÊN ==========\n\n"
    prompt += question

    prompt += (
        "\n\n========== YÊU CẦU TRẢ LỜI ==========\n\n"
        "Hãy trả lời theo ĐÚNG định dạng đã hướng dẫn ở trên:\n"
        "1. **Mở đầu**: tóm tắt 1-2 câu.\n"
        "2. **Thân bài**: chi tiết, có heading `###`, bullet list, "
        "**bold** từ khóa, dùng bảng nếu so sánh.\n"
        "3. **Kết luận**: tóm tắt hoặc gợi ý.\n\n"
        "RÀNG BUỘC:\n"
        "- Trả lời bằng TIẾNG VIỆT 100%.\n"
        "- Chỉ dùng thông tin trong CONTEXT.\n"
        "- Không bịa. Không nhầm ngành với chuyên ngành.\n"
        "- Không dùng từ Indonesia.\n"
        "- Độ dài: 300-500 từ cho câu hỏi phức tạp, "
        "150-300 từ cho câu hỏi đơn giản.\n\n"
        "BẮT ĐẦU TRẢ LỜI:\n"
    )
    return prompt


# ============================================================
# 13. HẬU XỬ LÝ — LỌC TỪ INDONESIA
# ============================================================

INDONESIAN_WORDS = {
    "jika": "nếu", "dan": "và", "untuk": "để",
    "dengan": "với", "adalah": "là", "tidak": "không",
    "bisa": "có thể", "akan": "sẽ", "saya": "tôi",
    "kamu": "bạn", "maka": "thì", "juga": "cũng",
    "sudah": "đã", "belum": "chưa", "atau": "hoặc",
    "karena": "vì", "tetapi": "nhưng", "namun": "tuy nhiên",
    "sangat": "rất", "lebih": "hơn", "paling": "nhất",
}


def clean_answer(answer: str) -> str:
    if not answer:
        return answer

    for foreign, viet in INDONESIAN_WORDS.items():
        pattern = r"\b" + re.escape(foreign) + r"\b"
        answer = re.sub(pattern, viet, answer, flags=re.IGNORECASE)

    answer = re.sub(r"[ \t]+", " ", answer)
    answer = re.sub(r"\n +", "\n", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)

    return answer.strip()


# ============================================================
# 14. OPTIONS
# ============================================================

def _build_options():
    opts = {
        "temperature": LLM_TEMPERATURE,
        "num_predict": LLM_NUM_PREDICT,
        "num_ctx": LLM_NUM_CTX,
    }
    if OLLAMA_NUM_THREAD:
        opts["num_thread"] = OLLAMA_NUM_THREAD
    return opts


# ============================================================
# 15. GỌI LLM
# ============================================================

def generate_answer(
    prompt,
    provider: str = "ollama",
    api_key: str = "",
    model: str = None,
):
    if provider == "ollama":
        try:
            response = ollama.chat(
                model=model or LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options=_build_options(),
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
        except Exception as e:
            raise RuntimeError(f"Lỗi khi gọi {model or LLM_MODEL}: {e}")

        answer = response.get("message", {}).get("content", "")
        if not answer:
            raise RuntimeError("Mô hình không trả về câu trả lời.")
        return clean_answer(answer.strip())

    if not api_key:
        raise RuntimeError(f"Thiếu API key cho provider '{provider}'.")

    chunks = []
    for piece in stream_external(prompt, provider, api_key, model=model):
        chunks.append(piece)
    return clean_answer("".join(chunks).strip())


def generate_answer_stream(
    prompt,
    provider: str = "ollama",
    api_key: str = "",
    model: str = None,
):
    if provider == "ollama":
        try:
            stream = ollama.chat(
                model=model or LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options=_build_options(),
                keep_alive=OLLAMA_KEEP_ALIVE,
                stream=True,
            )
        except Exception as e:
            raise RuntimeError(f"Lỗi khi gọi {model or LLM_MODEL}: {e}")

        for chunk in stream:
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
        return

    if not api_key:
        raise RuntimeError(f"Thiếu API key cho provider '{provider}'.")

    yield from stream_external(prompt, provider, api_key, model=model)


# ============================================================
# 16. API CẤP CAO
# ============================================================

def ask_ai(
    question,
    collection=None,
    top_k=FINAL_TOP_K,
    history=None,
    provider: str = "ollama",
    api_key: str = "",
    model: str = None,
):
    question = normalize_question(question)
    if not question:
        return "Vui lòng nhập câu hỏi.", []

    if collection is None:
        collection = get_chroma_collection()

    documents, metadatas, _ = hybrid_search_documents(
        collection, question, top_k
    )

    if not documents:
        return (
            "Hiện tại tôi chưa tìm thấy thông tin phù hợp "
            "trong kho dữ liệu.",
            [],
        )

    context = build_context(documents, metadatas, MAX_CONTEXT_CHARS)
    history = history[-6:] if history else None

    prompt = build_prompt(question, context, history)
    answer = generate_answer(
        prompt, provider=provider, api_key=api_key, model=model
    )
    return answer, metadatas


def prepare_rag_prompt(
    question,
    collection=None,
    top_k=FINAL_TOP_K,
    history=None,
):
    question = normalize_question(question)
    if not question:
        return None, []

    if collection is None:
        collection = get_chroma_collection()

    where = detect_doc_filter(question)
    documents, metadatas, _ = hybrid_search_documents(
        collection, question, top_k, where=where
    )

    if not documents and where:
        # Bộ lọc đoán sai / quá hẹp → thử lại không lọc thay vì bó tay.
        documents, metadatas, _ = hybrid_search_documents(
            collection, question, top_k, where=None
        )

    if not documents:
        return None, []

    context, used = build_context_ex(documents, metadatas, MAX_CONTEXT_CHARS)

    cited_metas = []
    for number, idx in enumerate(used, 1):
        meta = dict(metadatas[idx])
        meta["cite_no"] = number
        meta["snippet"] = make_snippet(documents[idx], query=question)
        cited_metas.append(meta)

    score_note = detect_score_conversion(question)
    if score_note:
        context = score_note + "\n\n" + context

    history = history[-6:] if history else None

    prompt = build_prompt(question, context, history)
    return prompt, cited_metas


# ============================================================
# 17. WARM-UP
# ============================================================

def warm_up(provider: str = "ollama"):
    ok = True

    if provider == "ollama":
        try:
            ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1, "num_ctx": 512},
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
        except Exception:
            ok = False

    if USE_RERANKER:
        try:
            get_reranker()
        except Exception:
            ok = False

    if USE_HYBRID:
        try:
            coll = get_chroma_collection()
            _build_bm25_index(coll)
        except Exception:
            ok = False

    return ok


# ============================================================
# 18. XÓA CACHE
# ============================================================

def clear_rag_cache():
    global _bm25_index, _bm25_corpus, _bm25_metadatas, _reranker, _embedder
    create_embedding.cache_clear()
    get_chroma_client.cache_clear()
    get_chroma_collection.cache_clear()
    _bm25_index = None
    _bm25_corpus = None
    _bm25_metadatas = None
    _reranker = None
    _embedder = None


# ============================================================
# 19. SMOKE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM TRA RAG_UTILS — HYBRID + RERANK + MULTI-LLM")
    print("=" * 60)
    try:
        collection = get_chroma_collection()
        print(f"✓ Collection : {COLLECTION_NAME}")
        print(f"✓ Embedding  : {EMBEDDING_MODEL}")
        print(f"✓ LLM local  : {LLM_MODEL}")
        print(f"✓ Số chunk   : {collection.count()}")
        print(f"✓ Hybrid     : {USE_HYBRID}")
        print(f"✓ Reranker   : {USE_RERANKER} ({RERANKER_MODEL})")
    except Exception as e:
        print(f"❌ Lỗi: {e}")