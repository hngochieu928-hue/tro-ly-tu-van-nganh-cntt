# ============================================================
# RAG.PY — CLI
# ============================================================

from rag_utils import (
    ask_ai,
    get_chroma_collection,
)

from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    TOP_K,
)


# ============================================================
# 1. HIỂN THỊ TIÊU ĐỀ
# ============================================================

def hien_thi_tieu_de():
    print()
    print("=" * 65)
    print("        TRỢ LÝ TƯ VẤN NGÀNH VÀ CHUYÊN NGÀNH")
    print("                 KHOA CNTT")
    print("=" * 65)
    print("Nhập 'exit' hoặc 'quit' để thoát.")
    print()


# ============================================================
# 2. HIỂN THỊ THÔNG TIN HỆ THỐNG
# ============================================================

def hien_thi_thong_tin():
    print(f"LLM        : {LLM_MODEL}")
    print(f"Embedding  : {EMBEDDING_MODEL}")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Top-K      : {TOP_K}")
    print()


# ============================================================
# 3. HIỂN THỊ NGUỒN
# ============================================================

def hien_thi_nguon(metadatas):
    if not metadatas:
        return

    print()
    print("--- NGUỒN TÀI LIỆU ---")

    da_hien_thi = set()

    for metadata in metadatas:
        source = metadata.get("source", "không rõ")
        chunk = metadata.get("chunk", "?")
        score = metadata.get("score")

        key = (source, chunk)
        if key in da_hien_thi:
            continue
        da_hien_thi.add(key)

        if score is not None:
            print(f"- {source} (chunk {chunk}, score {score:.4f})")
        else:
            print(f"- {source} (chunk {chunk})")


# ============================================================
# 4. MAIN
# ============================================================

def main():
    hien_thi_tieu_de()
    hien_thi_thong_tin()

    try:
        collection = get_chroma_collection()
    except Exception as e:
        print("❌ Không kết nối được ChromaDB.")
        print(f"Chi tiết: {e}")
        print()
        print("Hãy kiểm tra:")
        print("1. Đã chạy python build_db.py chưa?")
        print("2. DB_DIR trong config.py có đúng không?")
        print("3. Collection có tồn tại không?")
        return

    try:
        so_luong = collection.count()
    except Exception:
        so_luong = 0

    print(f"📚 Số lượng chunk trong database: {so_luong}")
    print()

    history = []

    while True:
        question = input("Sinh viên: ").strip()

        if question.lower() in ("exit", "quit"):
            print()
            print("Đã thoát chương trình.")
            break

        if not question:
            print("⚠️ Vui lòng nhập câu hỏi.")
            print()
            continue

        try:
            answer, metadatas = ask_ai(
                question,
                collection,
                TOP_K,
                history,
            )
        except Exception as e:
            answer = f"❌ Hệ thống gặp lỗi: {e}"
            metadatas = []

        print()
        print("=" * 65)
        print("TRỢ LÝ KHOA CNTT:")
        print("=" * 65)
        print(answer)

        hien_thi_nguon(metadatas)
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        if len(history) > 12:
            history = history[-12:]


# ============================================================
# 5. CHẠY
# ============================================================

if __name__ == "__main__":
    main()