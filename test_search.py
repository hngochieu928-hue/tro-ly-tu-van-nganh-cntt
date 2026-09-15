# ============================================================
# TEST_SEARCH.PY — KIỂM TRA HYBRID SEARCH + AUTO FILTER
# ============================================================
from config import (
    FINAL_TOP_K, CANDIDATES_K, EMBEDDING_MODEL, COLLECTION_NAME,
    RERANKER_MODEL, USE_HYBRID, USE_RERANKER,
)
from rag_utils import (
    get_chroma_collection, hybrid_search_documents,
    vector_search, bm25_search, detect_doc_filter,
)


def print_result(i, meta, score, doc, preview=400):
    source = meta.get("source", "?")
    chunk = meta.get("chunk", "?")
    doc_type = meta.get("doc_type", "?")
    kg_type = meta.get("kg_node_type", "")
    year = meta.get("year", "")

    quality = (
        "🟢 RẤT TỐT" if score > 0.7 else
        "🟡 TỐT" if score > 0.3 else
        "🟠 TẠM" if score > 0.0 else
        "🔴 YẾU"
    )

    print()
    print(f"--- KẾT QUẢ {i} ---")
    print(f"Nguồn    : {source}")
    print(f"Chunk    : {chunk}")
    print(f"Loại     : {doc_type}  {('· ' + kg_type) if kg_type else ''}")
    if year:
        print(f"Năm      : {year}")
    print(f"Score    : {score:.4f}  {quality}")
    print()
    print(doc[:preview])
    print("-" * 60)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    print("=" * 60)
    print("     KIỂM TRA HYBRID SEARCH + AUTO FILTER")
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
        print("⚠️  Câu hỏi trống.")
        return

    flt = detect_doc_filter(question)
    print(f"\n🔍 Auto Filter : {flt if flt else 'Không phát hiện'}")

    # 1. Vector
    section("1️⃣  VECTOR SEARCH")
    try:
        v_docs, v_metas, v_dists = vector_search(
            collection, question, CANDIDATES_K, where=flt
        )
        print(f"→ {len(v_docs)} chunk")
        for i, (m, d) in enumerate(zip(v_metas[:5], v_dists[:5]), 1):
            print(f"  [{i}] {m.get('source')} chunk {m.get('chunk')} "
                  f"[{m.get('doc_type')}] — dist {d:.4f}")
    except Exception as e:
        print(f"❌ {e}")

    # 2. BM25
    if USE_HYBRID and not flt:
        section("2️⃣  BM25 SEARCH")
        try:
            b_docs, b_metas = bm25_search(collection, question, CANDIDATES_K)
            print(f"→ {len(b_docs)} chunk")
            for i, m in enumerate(b_metas[:5], 1):
                print(f"  [{i}] {m.get('source')} chunk {m.get('chunk')} "
                      f"[{m.get('doc_type')}]")
        except Exception as e:
            print(f"❌ {e}")
    elif USE_HYBRID and flt:
        print("\n⚠️  Bỏ qua BM25 (đang có filter → chỉ dùng Vector Search)")

    # 3. Hybrid
    section("3️⃣  HYBRID + AUTO FILTER (KẾT QUẢ CUỐI)")
    try:
        docs, metas, scores = hybrid_search_documents(
            collection, question, FINAL_TOP_K, where=flt
        )
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return

    if not docs:
        print("⚠️  Không tìm thấy chunk nào.")
        return

    for i in range(min(len(docs), 5)):
        meta = metas[i] if i < len(metas) else {}
        score = scores[i] if i < len(scores) else 0.0
        print_result(i + 1, meta, score, docs[i])

    # 4. Filter test
    section("4️⃣  FILTER METADATA — KIỂM TRA NHANH")
    filter_tests = [
        {"doc_type": "quy_che"},
        {"doc_type": "tuyen_sinh"},
        {"doc_type": "diem_chuan"},
        {"doc_type": "quy_dinh", "category": "ktx"},
        {"doc_type": "quy_dinh", "category": "quy_doi_diem"},
        {"doc_type": "knowledge_graph"},
    ]
    for f in filter_tests:
        try:
            results = collection.get(where=f, include=["metadatas"])
            print(f"  {str(f):55s} → {len(results['ids']):4d} chunk")
        except Exception as e:
            print(f"  ❌ {f} lỗi: {e}")

    print("\n✅ Hoàn thành kiểm tra.")


if __name__ == "__main__":
    main()