# ============================================================
# BUILD_DB.PY
# ============================================================
# XÂY DỰNG VECTOR DATABASE
# ============================================================

import os
import hashlib

import ollama
import chromadb

from config import (
    DATA_DIR,
    DB_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ============================================================
# CẤU HÌNH BATCH
# ============================================================

EMBED_BATCH_SIZE = 32


# ============================================================
# ĐỌC FILE
# ============================================================

def read_file_auto_encoding(filepath):
    encodings = ("utf-8", "cp1258", "latin-1")

    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
        except OSError as e:
            print(f"❌ Lỗi đọc file: {e}")
            return None

    return None


# ============================================================
# CHUẨN HÓA TEXT
# ============================================================

def normalize_text(text):
    if not text:
        return ""

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


# ============================================================
# CHIA CHUNK — CẮT THEO DÒNG, GIỮ NGUYÊN NGỮ NGHĨA
# ============================================================

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Chia text thành các chunk theo RANH GIỚI DÒNG.
    Không bao giờ cắt giữa dòng → giữ nguyên ngữ nghĩa.
    Overlap được tính theo số dòng (không theo ký tự thô).
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("CHUNK_SIZE phải lớn hơn 0.")

    if overlap < 0:
        raise ValueError("CHUNK_OVERLAP không được âm.")

    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP phải nhỏ hơn CHUNK_SIZE.")

    lines = text.split("\n")

    chunks = []
    current_lines = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1

        if current_len + line_len > chunk_size and current_lines:
            chunk = "\n".join(current_lines).strip()
            if chunk:
                chunks.append(chunk)

            overlap_lines = []
            overlap_len = 0

            for old_line in reversed(current_lines):
                old_len = len(old_line) + 1
                if overlap_len + old_len > overlap:
                    break
                overlap_lines.insert(0, old_line)
                overlap_len += old_len

            current_lines = overlap_lines
            current_len = overlap_len

        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunk = "\n".join(current_lines).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


# ============================================================
# TẠO ID
# ============================================================

def create_document_id(filename, chunk_index):
    raw_id = f"{filename}_{chunk_index}"
    return hashlib.md5(raw_id.encode("utf-8")).hexdigest()


# ============================================================
# LẤY FILE TXT
# ============================================================

def get_text_files():
    if not os.path.exists(DATA_DIR):
        return []

    files = []
    for filename in os.listdir(DATA_DIR):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.isfile(filepath):
                files.append(filename)

    files.sort()
    return files


# ============================================================
# ĐỌC DỮ LIỆU
# ============================================================

def load_documents():
    documents = []
    metadatas = []
    ids = []
    file_count = 0
    chunk_count = 0

    files = get_text_files()

    if not files:
        print(f"❌ Không tìm thấy file TXT trong:")
        print(DATA_DIR)
        return (documents, metadatas, ids, file_count, chunk_count)

    print(f"📁 Tìm thấy {len(files)} file TXT.")
    print()

    for filename in files:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"📄 {filename}")

        text = read_file_auto_encoding(filepath)
        if text is None:
            print("   ⚠️ Không đọc được.")
            continue

        text = normalize_text(text)
        if not text:
            print("   ⚠️ File rỗng.")
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue

        file_count += 1

        for index, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk": index})
            ids.append(create_document_id(filename, index))
            chunk_count += 1

        print(f"   → {len(chunks)} chunk")

    return (documents, metadatas, ids, file_count, chunk_count)


# ============================================================
# TẠO EMBEDDING THEO BATCH
# ============================================================

def create_embeddings(documents):
    embeddings = []
    total = len(documents)

    if total == 0:
        return []

    print()
    print(f"🧠 Model: {EMBEDDING_MODEL}")
    print(f"📄 Tổng chunk: {total}")
    print(f"📦 Batch: {EMBED_BATCH_SIZE}")
    print()

    start = 0
    while start < total:
        end = min(start + EMBED_BATCH_SIZE, total)
        batch = documents[start:end]

        batch_number = (start // EMBED_BATCH_SIZE) + 1
        total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

        print(f"🔄 Batch {batch_number}/{total_batches}")

        try:
            response = ollama.embed(
                model=EMBEDDING_MODEL,
                input=batch,
            )
        except Exception as e:
            raise RuntimeError(
                f"Lỗi embedding batch {batch_number}: {e}"
            )

        batch_embeddings = response.get("embeddings")

        if not batch_embeddings:
            raise RuntimeError("Ollama không trả về embedding.")

        if len(batch_embeddings) != len(batch):
            raise RuntimeError(
                "Số embedding không khớp với số document."
            )

        embeddings.extend(batch_embeddings)
        start = end

    print()
    return embeddings


# ============================================================
# KẾT NỐI CHROMADB
# ============================================================

def get_chroma_client():
    os.makedirs(DB_DIR, exist_ok=True)
    print(f"🗄️ DB: {DB_DIR}")
    client = chromadb.PersistentClient(path=DB_DIR)
    return client


# ============================================================
# TẠO COLLECTION
# ============================================================

def create_collection(client):
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("🗑️ Đã xóa collection cũ.")
    except Exception:
        print("ℹ️ Không có collection cũ.")

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )
    return collection


# ============================================================
# LƯU DATABASE
# ============================================================

def save_to_database(
    collection, ids, documents, metadatas, embeddings
):
    total = len(ids)
    if total == 0:
        return

    batch_size = 100
    start = 0

    while start < total:
        end = min(start + batch_size, total)

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

        print(f"💾 Đã lưu {end}/{total}")
        start = end


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 65)
    print("       XÂY DỰNG VECTOR DATABASE")
    print("       TRỢ LÝ KHOA CNTT")
    print("=" * 65)
    print()

    if not os.path.exists(DATA_DIR):
        print("❌ Không tồn tại thư mục data:")
        print(DATA_DIR)
        return

    (documents, metadatas, ids, file_count, chunk_count) = load_documents()

    if not documents:
        print("❌ Không có dữ liệu.")
        return

    print()
    print("=" * 65)
    print("📊 THỐNG KÊ")
    print(f"File : {file_count}")
    print(f"Chunk: {chunk_count}")
    print("=" * 65)

    try:
        embeddings = create_embeddings(documents)
    except Exception as e:
        print(f"❌ {e}")
        return

    if len(embeddings) != len(documents):
        print("❌ Embedding không khớp.")
        return

    try:
        client = get_chroma_client()
        collection = create_collection(client)
    except Exception as e:
        print(f"❌ Lỗi ChromaDB: {e}")
        return

    try:
        save_to_database(
            collection, ids, documents, metadatas, embeddings
        )
    except Exception as e:
        print(f"❌ Lỗi lưu database: {e}")
        return

    print()
    print(f"📦 Collection: {COLLECTION_NAME}")
    print(f"🔢 Số chunk trong DB: {collection.count()}")
    print()
    print("=" * 65)
    print("✅ XÂY DỰNG DATABASE THÀNH CÔNG")
    print("=" * 65)
    print()
    print("👉 Tiếp theo chạy:")
    print("   streamlit run app.py")
    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()