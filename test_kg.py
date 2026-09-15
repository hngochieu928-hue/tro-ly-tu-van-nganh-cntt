# ============================================================
# TEST_KG.PY — KIỂM TRA CHẤT LƯỢNG ĐỒ THỊ TRI THỨC
# ============================================================
import os
import re

from config import KG_DIR
from kg_schema import NODE_TYPES, RELATIONSHIP_TYPES


FILE_TO_NODE = {
    "kg_university.txt":          "University",
    "kg_major.txt":               "Major",
    "kg_university_major.txt":    "UniversityMajor",
    "kg_admission_criteria.txt":  "AdmissionCriteria",
    "kg_subject_combination.txt": "SubjectCombination",
    "kg_admission_method.txt":    "AdmissionMethod",
    "kg_major_statistic.txt":     "MajorStatistic",
    "kg_program_type.txt":        "ProgramType",
}


KEYWORD_CHECKS = {
    "University":         ["Trường Đại học Điện lực", "DDL"],
    "Major":              ["mã ngành", "ngành"],
    "UniversityMajor":    ["tuyển sinh", "chỉ tiêu", "năm"],
    "AdmissionCriteria":  ["điểm chuẩn", "thang điểm"],
    "SubjectCombination": ["Tổ hợp", "môn"],
    "AdmissionMethod":    ["Phương thức", "ĐXT"],
    "MajorStatistic":     ["việc làm", "tỷ lệ"],
    "ProgramType":        ["chương trình", "đào tạo"],
}


def count_docs(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return [d.strip() for d in content.split("\n\n") if d.strip()]


def validate_doc(node_type, text):
    keywords = KEYWORD_CHECKS.get(node_type, [])
    if not keywords:
        return True
    return any(kw.lower() in text.lower() for kw in keywords)


def check_schema():
    print("📐 KIỂM TRA LƯỢC ĐỒ")
    print("-" * 70)

    n_nodes = len(NODE_TYPES)
    n_rels = len(RELATIONSHIP_TYPES)

    ok_nodes = "✅" if n_nodes == 8 else "⚠️"
    ok_rels = "✅" if n_rels == 7 else "⚠️"

    print(f"   {ok_nodes} Số loại nút       : {n_nodes}  (kỳ vọng: 8)")
    print(f"   {ok_rels} Số loại quan hệ   : {n_rels}  (kỳ vọng: 7)")

    print("\n   Kiểm tra quan hệ (from → to):")
    for rel, spec in RELATIONSHIP_TYPES.items():
        src, dst = spec["from"], spec["to"]
        valid_src = src in NODE_TYPES
        valid_dst = dst in NODE_TYPES
        mark = "✅" if (valid_src and valid_dst) else "❌"
        print(f"      {mark} {rel:25s} : {src:20s} → {dst}")
    print()


def check_content(files):
    print("🔍 KIỂM TRA NỘI DUNG")
    print("-" * 70)
    print(f"   {'File':35s} {'Loại':22s} {'Số đoạn':>8s} {'Hợp lệ':>8s} {'Tỉ lệ':>8s}")
    print("   " + "-" * 66)

    total_docs = 0
    total_ok = 0
    missing_types = []

    for filename in files:
        node_type = FILE_TO_NODE.get(filename, "Unknown")
        if node_type == "Unknown":
            continue

        filepath = os.path.join(KG_DIR, filename)
        docs = count_docs(filepath)
        if not docs:
            continue

        ok = sum(1 for d in docs if validate_doc(node_type, d))
        rate = (ok / len(docs) * 100) if docs else 0

        total_docs += len(docs)
        total_ok += ok

        print(f"   {filename:35s} {node_type:22s} {len(docs):>8d} {ok:>8d} {rate:>7.1f}%")

    for node_type in NODE_TYPES.keys():
        found = any(FILE_TO_NODE.get(f) == node_type for f in files)
        if not found:
            missing_types.append(node_type)

    print("   " + "-" * 66)
    rate = (total_ok / total_docs * 100) if total_docs else 0
    print(f"   {'TỔNG':35s} {'':22s} {total_docs:>8d} {total_ok:>8d} {rate:>7.1f}%")
    print()

    if missing_types:
        print(f"   ⚠️  Loại nút chưa có file:")
        for nt in missing_types:
            print(f"      - {nt}")
        print()

    return total_docs, total_ok


def check_size():
    print("📏 KÍCH THƯỚC FILE")
    print("-" * 70)

    total_size = 0
    for filename in sorted(os.listdir(KG_DIR)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(KG_DIR, filename)
        size = os.path.getsize(filepath)
        total_size += size
        print(f"   {filename:42s} {size/1024:>8.1f} KB")

    print(f"   {'TỔNG':42s} {total_size/1024:>8.1f} KB")
    print()


def main():
    print()
    print("=" * 70)
    print("   KIỂM TRA CHẤT LƯỢNG ĐỒ THỊ TRI THỨC")
    print("=" * 70)
    print()

    check_schema()

    print("📁 KIỂM TRA FILE KG")
    print("-" * 70)

    if not os.path.exists(KG_DIR):
        print(f"   ❌ Chưa có thư mục: {KG_DIR}")
        print(f"   👉 Chạy: python build_kg.py")
        return

    files = sorted(f for f in os.listdir(KG_DIR) if f.endswith(".txt"))
    if not files:
        print(f"   ❌ Chưa có file KG nào trong {KG_DIR}")
        print(f"   👉 Chạy: python build_kg.py")
        return

    print(f"   ✅ Tìm thấy {len(files)} file KG\n")

    result = check_content(files)
    if not result:
        return
    total_docs, total_ok = result

    check_size()

    print("=" * 70)
    rate = total_ok / total_docs if total_docs else 0
    if rate >= 0.85:
        print("✅ CHẤT LƯỢNG ĐỒ THỊ TRI THỨC: TỐT")
    elif rate >= 0.6:
        print("🟡 CHẤT LƯỢNG ĐỒ THỊ TRI THỨC: TRUNG BÌNH")
    else:
        print("🔴 CHẤT LƯỢNG ĐỒ THỊ TRI THỨC: CẦN CẢI THIỆN")
    print(f"   Tổng đoạn  : {total_docs}")
    print(f"   Hợp lệ     : {total_ok} ({rate*100:.1f}%)")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()