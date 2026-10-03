# ============================================================
# BUILD_DB.PY
# ============================================================
# XÂY DỰNG VECTOR DATABASE
# ============================================================

import os
import re
import hashlib

import chromadb

from config import (
    DATA_DIR,
    DB_DIR,
    KG_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


# ============================================================
# PHÂN LOẠI VĂN BẢN (metadata cho auto-filter khi tìm kiếm)
# ============================================================

FILE_TAGS = [
    (r"^Diem_chuan_nam_", {"doc_type": "diem_chuan"}),
    (r"^Thong tin tuyen sinh nam ", {"doc_type": "tuyen_sinh"}),
    (r"^Quy che dao tao", {"doc_type": "quy_che"}),
    (r"^Quy dinh muc thu tien KTX", {"doc_type": "quy_dinh", "category": "ktx"}),
    (r"^Muc thu tien KTX", {"doc_type": "quy_dinh", "category": "ktx"}),
    (r"^Muc thu hoc phi", {"doc_type": "quy_dinh", "category": "hoc_phi"}),
    (r"^Chinh sach mien giam hoc phi", {"doc_type": "quy_dinh", "category": "mien_giam_hoc_phi"}),
    (r"^Quy dinh xet cap hoc bong", {"doc_type": "quy_dinh", "category": "hoc_bong"}),
    (r"^Quy trinh ky luat", {"doc_type": "quy_dinh", "category": "ky_luat"}),
    (r"^Quy dinh dong phuc", {"doc_type": "quy_dinh", "category": "dong_phuc"}),
    (r"^Tieu chi danh gia ket qua ren luyen", {"doc_type": "quy_dinh", "category": "ren_luyen"}),
    (r"^Quy_dinh_quy_doi_diem_hoc_ba", {"doc_type": "quy_dinh", "category": "quy_doi_diem"}),
    (r"^chuong_trinh_khung", {"doc_type": "chuong_trinh"}),
    (r"^gioi_thieu_nganh", {"doc_type": "gioi_thieu"}),
    # Tài liệu tư vấn định hướng (chia đoạn theo mục, xem chunk_sections)
    (r"^dinh_huong_nghe_nghiep", {"doc_type": "tu_van", "category": "nghe_nghiep"}),
    (r"^ky_nang_theo_nganh", {"doc_type": "tu_van", "category": "ky_nang"}),
    (r"^so_sanh_nganh", {"doc_type": "tu_van", "category": "so_sanh"}),
    (r"^lo_trinh_tu_van", {"doc_type": "tu_van", "category": "lo_trinh"}),
]

KG_FILE_TO_NODE = {
    "kg_university.txt":          "University",
    "kg_major.txt":               "Major",
    "kg_university_major.txt":    "UniversityMajor",
    "kg_admission_criteria.txt":  "AdmissionCriteria",
    "kg_subject_combination.txt": "SubjectCombination",
    "kg_admission_method.txt":    "AdmissionMethod",
    "kg_major_statistic.txt":     "MajorStatistic",
    "kg_program_type.txt":        "ProgramType",
}


def classify_doc(filename):
    meta = {"doc_type": "khac"}
    for pattern, tags in FILE_TAGS:
        if re.match(pattern, filename):
            meta.update(tags)
            break

    year_match = re.search(r"20\d{2}", filename)
    if year_match:
        meta["year"] = year_match.group()

    return meta


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
# CHIA ĐOẠN THEO MỤC (cho tài liệu tư vấn có tiêu đề ## / ###)
# ============================================================
# Các tệp như ky_nang_theo_nganh.txt lặp lại cùng một tiêu đề con ("Có học AI
# không?", "Mức độ lập trình"...) cho từng chuyên ngành. Nếu chia đoạn theo ký
# tự, một đoạn có thể mất dòng tên chuyên ngành và câu trả lời bị gán nhầm cho
# chuyên ngành khác. Vì vậy: (1) mỗi đoạn được gắn tiêu đề tài liệu + tên mục;
# (2) bảng markdown được chuyển thành các câu đủ ngữ nghĩa theo từng hàng.

def _clean_md(line):
    line = line.strip()
    line = re.sub(r"^#{1,6}\s*", "", line)
    return line.replace("**", "")


def _is_table_line(line):
    return line.strip().startswith("|")


def _table_cells(line):
    return [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]


def linearize_tables(lines):
    """Bảng markdown → mỗi hàng thành một câu: «Cột 1 'giá trị' — Cột 2: ...; Cột 3: ...»."""
    out, i = [], 0
    while i < len(lines):
        if (_is_table_line(lines[i]) and i + 1 < len(lines)
                and re.match(r"^\|?\s*:?-{3,}", lines[i + 1].strip().lstrip("|").strip())):
            header = _table_cells(lines[i])
            i += 2
            while i < len(lines) and _is_table_line(lines[i]):
                cells = _table_cells(lines[i])
                head = f"{header[0]} '{cells[0]}'" if cells else ""
                rest = "; ".join(
                    f"{header[k]}: {cells[k]}"
                    for k in range(1, min(len(header), len(cells)))
                )
                out.append(f"{head} — {rest}." if rest else f"{head}.")
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def _split_blocks(lines):
    """Tách thành khối nhỏ (### hoặc dòng **in đậm** đứng một mình) để không cắt giữa khối."""
    blocks, cur = [], []
    for ln in lines:
        s = ln.strip()
        starts_block = s.startswith("### ") or (s.startswith("**") and s.endswith("**") and len(s) > 4)
        if starts_block and cur:
            blocks.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        blocks.append(cur)
    return blocks


def chunk_sections(text, chunk_size=CHUNK_SIZE, overlap=150):
    lines = text.split("\n")
    title = ""
    sections, heading, buf = [], "", []
    for ln in lines:
        s = ln.strip()
        if not s or set(s) <= {"="}:
            continue
        if s.startswith("## "):
            if buf:
                sections.append((heading, buf))
            heading, buf = _clean_md(s), []
        elif not title and not s.startswith("#"):
            title = _clean_md(s)
        else:
            buf.append(ln)
    if buf:
        sections.append((heading, buf))

    chunks = []
    for heading, body in sections:
        prefix = f"[{title}] {heading}\n" if heading else f"[{title}]\n"
        room = max(200, chunk_size - len(prefix))
        body = linearize_tables(body)
        current, cur_len = [], 0
        for block in _split_blocks(body):
            text_block = "\n".join(_clean_md(x) for x in block)
            if len(text_block) > room:       # khối quá dài → chia theo dòng
                if current:
                    chunks.append(prefix + "\n".join(current))
                    current, cur_len = [], 0
                for piece in chunk_text(text_block, room, overlap):
                    chunks.append(prefix + piece)
                continue
            if current and cur_len + len(text_block) + 1 > room:
                chunks.append(prefix + "\n".join(current))
                current, cur_len = [], 0
            current.append(text_block)
            cur_len += len(text_block) + 1
        if current:
            chunks.append(prefix + "\n".join(current))
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

def get_kg_files():
    if not os.path.exists(KG_DIR):
        return []

    files = []
    for filename in os.listdir(KG_DIR):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(KG_DIR, filename)
            if os.path.isfile(filepath):
                files.append(filename)

    files.sort()
    return files


def load_documents():
    documents = []
    metadatas = []
    ids = []
    file_count = 0
    chunk_count = 0

    files = get_text_files()
    kg_files = get_kg_files()

    if not files and not kg_files:
        print(f"❌ Không tìm thấy file TXT trong:")
        print(DATA_DIR)
        return (documents, metadatas, ids, file_count, chunk_count)

    print(f"📁 Tìm thấy {len(files)} file TXT + {len(kg_files)} file KG.")
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

        tags = classify_doc(filename)
        if tags.get("doc_type") == "tu_van":
            chunks = chunk_sections(text)
        else:
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue

        file_count += 1

        for index, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk": index, **tags})
            ids.append(create_document_id(filename, index))
            chunk_count += 1

        print(f"   → {len(chunks)} chunk")

    for filename in kg_files:
        filepath = os.path.join(KG_DIR, filename)
        print(f"📄 [KG] {filename}")

        text = read_file_auto_encoding(filepath)
        if text is None:
            print("   ⚠️ Không đọc được.")
            continue

        # Mỗi đoạn (ngăn cách bởi dòng trống) là một sự thật nguyên tử đã
        # được sinh sẵn từ đồ thị tri thức — không cắt lại theo ký tự.
        facts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not facts:
            print("   ⚠️ File rỗng.")
            continue

        file_count += 1
        kg_node_type = KG_FILE_TO_NODE.get(filename, "Unknown")

        for index, fact in enumerate(facts):
            documents.append(fact)
            metadatas.append({
                "source": filename,
                "chunk": index,
                "doc_type": "knowledge_graph",
                "kg_node_type": kg_node_type,
            })
            ids.append(create_document_id(filename, index))
            chunk_count += 1

        print(f"   → {len(facts)} chunk")

    return (documents, metadatas, ids, file_count, chunk_count)


# ============================================================
# TẠO EMBEDDING THEO BATCH
# ============================================================

def create_embeddings(documents):
    total = len(documents)

    if total == 0:
        return []

    print()
    print(f"🧠 Model: {EMBEDDING_MODEL}")
    print(f"📄 Tổng chunk: {total}")
    print(f"📦 Batch: {EMBED_BATCH_SIZE}")
    print()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)

    vectors = model.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print()
    return vectors.tolist()


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