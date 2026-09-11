# ============================================================
# TEST_SEARCH.PY — KIỂM TRA HYBRID + RERANK
# ============================================================

from config import (
    FINAL_TOP_K,
    CANDIDATES_K,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    RERANKER_MODEL,
    USE_HYBRID,
    USE_RERANKER,
)

from rag_utils import (
    get_chroma_collection,
    hybrid_search_documents,
    vector_search,
    bm25_search,
)


def main():
    print("=" * 60)
    print("     KIỂM TRA HYBRID SEARCH + RERANKER")
    print("=" * 60)

    try:
        collection = get_chroma_collection()
    except Exception as e:
        print(f"❌ Lỗi kết nối ChromaDB: {e}")
        return

    print(f"📦 Collection  : {COLLECTION_NAME}")
    print(f"🧠 Embedding   : {EMBEDDING_MODEL}")
    print(f"🎯 Reranker    : {RERANKER_MODEL if USE_RERANKER else 'OFF'}")
    print(f"🔀 Hybrid BM25 : {'ON' if USE_HYBRID else 'OFF'}")
    print(f"📊 Ứng viên    : {CANDIDATES_K} → {FINAL_TOP_K}")
    print(f"📄 Số chunk    : {collection.count()}")
    print()

    question = input("Nhập câu hỏi kiểm tra: ").strip()
    if not question:
        print("⚠️ Câu hỏi trống.")
        return

    # ============ 1. VECTOR ============
    print("\n" + "=" * 60)
    print("1️⃣  VECTOR SEARCH")
    print("=" * 60)

    try:
        v_docs, v_metas, v_dists = vector_search(
            collection, question, CANDIDATES_K
        )
        print(f"→ {len(v_docs)} chunk")
        for i, (m, d) in enumerate(zip(v_metas[:3], v_dists[:3])):
            print(f"  [{i+1}] {m.get('source')} "
                  f"chunk {m.get('chunk')} — dist {d:.4f}")
    except Exception as e:
        print(f"❌ {e}")

    # ============ 2. BM25 ============
    if USE_HYBRID:
        print("\n" + "=" * 60)
        print("2️⃣  BM25 SEARCH")
        print("=" * 60)
        try:
            b_docs, b_metas = bm25_search(
                collection, question, CANDIDATES_K
            )
            print(f"→ {len(b_docs)} chunk")
            for i, m in enumerate(b_metas[:3]):
                print(f"  [{i+1}] {m.get('source')} "
                      f"chunk {m.get('chunk')}")
        except Exception as e:
            print(f"❌ {e}")

    # ============ 3. HYBRID + RERANK ============
    print("\n" + "=" * 60)
    print("3️⃣  HYBRID + RERANK (KẾT QUẢ CUỐI)")
    print("=" * 60)

    try:
        docs, metas, scores = hybrid_search_documents(
            collection, question, FINAL_TOP_K
        )
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return

    if not docs:
        print("⚠️ Không tìm thấy chunk nào.")
        return

    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else {}
        score = scores[i] if i < len(scores) else 0.0

        source = meta.get("source", "?")
        chunk = meta.get("chunk", "?")

        if score > 0.7:
            quality = "🟢 RẤT TỐT"
        elif score > 0.3:
            quality = "🟡 TỐT"
        elif score > 0.0:
            quality = "🟠 TẠM"
        else:
            quality = "🔴 YẾU"

        print()
        print(f"--- KẾT QUẢ {i + 1} ---")
        print(f"Nguồn    : {source}")
        print(f"Chunk    : {chunk}")
        print(f"Score    : {score:.4f}  {quality}")
        print()
        print(docs[i][:500])
        print("-" * 60)

    print("\n✅ Hoàn thành kiểm tra.")


if __name__ == "__main__":
    main()