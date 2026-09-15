# ============================================================
# BUILD_KG.PY — CHUYỂN DỮ LIỆU THÀNH ĐỒ THỊ TRI THỨC
# ============================================================
import os
import re

from config import DATA_DIR, KG_DIR
from kg_schema import TEXT_TEMPLATES


def read_file_auto_encoding(filepath):
    for enc in ("utf-8", "cp1258", "latin-1"):
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def load_raw_data():
    raw = {}
    if not os.path.exists(DATA_DIR):
        return raw

    for fn in os.listdir(DATA_DIR):
        fp = os.path.join(DATA_DIR, fn)
        if not os.path.isfile(fp) or not fn.lower().endswith(".txt"):
            continue
        txt = read_file_auto_encoding(fp)
        if txt:
            raw[fn] = txt
    return raw


# ============================================================
# TRÍCH XUẤT THỰC THỂ
# ============================================================

def extract_university(raw_data):
    univ = {
        "name": "Trường Đại học Điện lực",
        "code": "DDL",
        "type": "công lập trực thuộc Bộ Công Thương",
        "address": (
            "Cơ sở 1: Số 235 Hoàng Quốc Việt, phường Nghĩa Đô, Hà Nội; "
            "Cơ sở 2: Xã Sóc Sơn, Hà Nội; "
            "Cơ sở 3: 126 Phố Xóm, phường Phú Lương, Hà Nội; "
            "Cơ sở 4: Khu Công nghệ cao Hòa Lạc, Hà Nội"
        ),
        "contact": "Điện thoại 024 224 52662; Email tuyensinh@epu.edu.vn",
        "website": "https://epu.edu.vn hoặc https://tuyensinh.epu.edu.vn",
    }

    for fn, txt in raw_data.items():
        if "mã cơ sở đào tạo" in txt.lower():
            m = re.search(
                r"mã\s+cơ sở đào tạo[^\n:]*[:\s]+([A-Z]{2,5})", txt
            )
            if m:
                univ["code"] = m.group(1).strip()

    return [univ]


def extract_majors(raw_data):
    majors, seen = [], set()

    for fn, txt in raw_data.items():
        if "tuyen sinh" not in fn.lower() and "diem_chuan" not in fn.lower():
            continue

        for m in re.finditer(r"\d+\s*\|\s*(\d{7})\s*\|\s*([^|\n]+?)\s*\|", txt):
            code, name = m.group(1).strip(), m.group(2).strip()
            if code in seen or len(name) < 5:
                continue
            seen.add(code)
            majors.append({
                "code": code, "name": name,
                "description": "Chương trình đào tạo trình độ đại học chính quy",
                "career": "Cơ hội nghề nghiệp đa dạng theo định hướng ngành",
            })
    return majors


def extract_subject_combinations(raw_data):
    combos, seen = [], set()

    for fn, txt in raw_data.items():
        if "tuyen sinh nam 2026" not in fn.lower():
            continue

        patterns = [
            r"\b([A-DX]\d{2})\s*\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|\s*([^|\n]+)",
            r"\d+\.\s*([A-DX]\d{2})\s+([A-ZÀ-Ỹ][^\n]+?)\s+([A-ZÀ-Ỹ][^\n]+?)\s+([A-ZÀ-Ỹ][^\n]+)",
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, txt):
                code = m.group(1).strip()
                if code in seen:
                    continue
                seen.add(code)
                combos.append({
                    "code": code,
                    "subject1": m.group(2).strip(),
                    "subject2": m.group(3).strip(),
                    "subject3": m.group(4).strip(),
                    "description": f"Tổ hợp {code}",
                })
    return combos


def extract_admission_methods(raw_data):
    return [
        {
            "name": "Phương thức 1 — Xét tuyển dựa trên học bạ THPT",
            "description": (
                "Sử dụng kết quả học tập cả năm lớp 10, 11, 12 của 3 môn "
                "theo tổ hợp xét tuyển và điểm ưu tiên (nếu có). "
                "Điểm xét tuyển được quy đổi về phương thức gốc THPT "
                "theo công thức nội suy tuyến tính: "
                "y = a + (x - m) / (n - m) × (b - a), trong đó x là tổng "
                "điểm học bạ thực tế, [m, n] là khoảng điểm học bạ theo mốc "
                "phân vị, [a, b] là khoảng điểm tương ứng của phương thức THPT."
            ),
            "formula": (
                "ĐXT = (ĐTB Môn 1 + ĐTB Môn 2 + ĐTB Môn 3) + ĐUT. "
                "Quy đổi sang phương thức gốc THPT: "
                "y = a + (x - m) / (n - m) × (b - a)"
            ),
        },
        {
            "name": "Phương thức 2 — Xét tuyển kết hợp chứng chỉ TAQT với học bạ",
            "description": (
                "Sử dụng chứng chỉ IELTS hoặc TOEFL iBT kết hợp với kết quả "
                "học tập 02 môn còn lại trong tổ hợp. Điểm cũng được quy đổi "
                "về phương thức gốc THPT theo cùng công thức nội suy."
            ),
            "formula": (
                "ĐXT = (Điểm CCTA quy đổi + ĐTB Môn 1 + ĐTB Môn 2) + ĐUT. "
                "Quy đổi: y = a + (x - m) / (n - m) × (b - a)"
            ),
        },
        {
            "name": "Phương thức 3 — Xét tuyển dựa trên điểm thi tốt nghiệp THPT",
            "description": (
                "Sử dụng kết quả điểm trong kỳ thi tốt nghiệp THPT của "
                "3 môn theo tổ hợp xét tuyển. Đây là phương thức gốc, "
                "không cần quy đổi."
            ),
            "formula": "ĐXT = Tổng điểm 3 môn TN THPT + ĐUT",
        },
        {
            "name": "Phương thức 4 — Xét tuyển thẳng",
            "description": (
                "Xét tuyển thẳng theo Điều 8 Thông tư 06/2026/TT-BGDĐT. "
                "Không áp dụng công thức quy đổi."
            ),
            "formula": "Theo quy định của Bộ GD&ĐT",
        },
    ]


def extract_admission_criteria(raw_data):
    criteria = []

    for fn, txt in raw_data.items():
        if "diem_chuan" not in fn.lower():
            continue

        ym = re.search(r"20\d{2}", fn)
        year = ym.group() if ym else "unknown"

        pattern = r"\d+\s*\|\s*(\d{7})\s*\|\s*([^|\n]+?)\s*\|\s*([\d,\.]+)"
        for m in re.finditer(pattern, txt):
            code = m.group(1).strip()
            name = m.group(2).strip()
            try:
                score = float(m.group(3).strip().replace(",", "."))
            except ValueError:
                continue

            criteria.append({
                "criteriaID": f"EPU_{code}_{year}",
                "year": year,
                "major_code": code,
                "major_name": name,
                "entranceScore": score,
                "quota": "Theo đề án tuyển sinh",
            })
    return criteria


def extract_major_statistics(raw_data):
    # TẠM TẮT: regex cũ bắt sai cột (khớp nhầm số thứ tự/chỉ tiêu thành tỉ lệ
    # việc làm, sinh ra số liệu vô lý như 5-30% trong khi bảng gốc thực tế là
    # 78-100%). Dữ liệu bảng việc làm vẫn được RAG lấy đúng từ chunk văn bản
    # thô của "Thong tin tuyen sinh nam 2024.txt", nên bỏ qua lớp KG này cho
    # đến khi có thời gian viết lại regex và đối chiếu kỹ với bảng gốc.
    return []


def extract_program_types():
    return [
        {"name": "Cử nhân", "description": "Chương trình đào tạo trình độ đại học cấp bằng cử nhân",
         "features": "Thời gian 4 năm, 8 học kỳ chính"},
        {"name": "Kỹ sư", "description": "Chương trình đào tạo trình độ đại học cấp bằng kỹ sư",
         "features": "Thời gian 4,5 năm, 9 học kỳ chính"},
        {"name": "Chương trình chuẩn", "description": "Chương trình đào tạo theo chuẩn của Bộ GD&ĐT",
         "features": "Đảm bảo chuẩn đầu ra theo Khung trình độ quốc gia VN"},
        {"name": "Chương trình chất lượng cao", "description": "Chương trình đào tạo nâng cao, tăng cường tiếng Anh",
         "features": "Yêu cầu đầu vào cao hơn chương trình chuẩn"},
        {"name": "Chương trình vừa làm vừa học", "description": "Chương trình đào tạo dành cho người đã đi làm",
         "features": "Thời gian đào tạo dài hơn tối thiểu 20% so với chính quy"},
        {"name": "Chương trình liên thông", "description": "Chương trình đào tạo liên thông từ TC/CĐ lên đại học",
         "features": "Công nhận kết quả học tập đã tích lũy theo quy định"},
    ]


def build_university_major(universities, majors, criteria):
    if not universities or not majors:
        return []

    univ = universities[0]
    major_map = {m["code"]: m["name"] for m in majors}

    by_year_major = {}
    for c in criteria:
        by_year_major[(c["major_code"], c["year"])] = c

    ums = []
    for (code, year), c in by_year_major.items():
        ums.append({
            "university_code": univ["code"],
            "university_name": univ["name"],
            "major_code": code,
            "major_name": major_map.get(code, c["major_name"]),
            "year": year,
            "tuition": "Theo quy định (420.000 - 590.000 đồng/tín chỉ)",
            "duration": "4 năm (cử nhân) hoặc 4,5 năm (kỹ sư)",
            "quota": c["quota"],
        })
    return ums


# ============================================================
# SINH VĂN BẢN
# ============================================================

def safe_format(template, **kwargs):
    try:
        return template.format(**kwargs)
    except (KeyError, IndexError):
        for key in re.findall(r"\{(\w+)\}", template):
            if key not in kwargs:
                template = template.replace(f"{{{key}}}", "N/A")
        try:
            return template.format(**kwargs)
        except Exception:
            return template


def gen_text(node_type, items):
    tmpl = TEXT_TEMPLATES[node_type]
    return [safe_format(tmpl, **item) for item in items]


def generate_all_texts(kg):
    methods = kg.get("AdmissionMethod", [])

    method_names = "; ".join(m["name"] for m in methods[:3])

    criteria_texts = [
        safe_format(
            TEXT_TEMPLATES["AdmissionCriteria"],
            year=c["year"], major_name=c["major_name"],
            entranceScore=c["entranceScore"], quota=c["quota"],
            method_names=method_names or "Theo đề án",
        )
        for c in kg["AdmissionCriteria"]
    ]

    return {
        "University":         gen_text("University", kg["University"]),
        "Major":              gen_text("Major", kg["Major"]),
        "UniversityMajor":    gen_text("UniversityMajor", kg["UniversityMajor"]),
        "AdmissionCriteria":  criteria_texts,
        "SubjectCombination": gen_text("SubjectCombination", kg["SubjectCombination"]),
        "AdmissionMethod":    gen_text("AdmissionMethod", kg["AdmissionMethod"]),
        "MajorStatistic":     gen_text("MajorStatistic", kg["MajorStatistic"]),
        "ProgramType":        gen_text("ProgramType", kg["ProgramType"]),
    }


# ============================================================
# LƯU
# ============================================================

FILE_MAP = {
    "University":         "kg_university.txt",
    "Major":              "kg_major.txt",
    "UniversityMajor":    "kg_university_major.txt",
    "AdmissionCriteria":  "kg_admission_criteria.txt",
    "SubjectCombination": "kg_subject_combination.txt",
    "AdmissionMethod":    "kg_admission_method.txt",
    "MajorStatistic":     "kg_major_statistic.txt",
    "ProgramType":        "kg_program_type.txt",
}


def save_kg_texts(texts_by_type):
    os.makedirs(KG_DIR, exist_ok=True)
    for fn in os.listdir(KG_DIR):
        if fn.endswith(".txt"):
            os.remove(os.path.join(KG_DIR, fn))

    for node_type, texts in texts_by_type.items():
        if not texts:
            continue
        filename = FILE_MAP.get(node_type, f"kg_{node_type.lower()}.txt")
        with open(os.path.join(KG_DIR, filename), "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts))
        print(f"   💾 {filename:42s} ({len(texts):4d} đoạn)")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 65)
    print("   CHUYỂN DỮ LIỆU EPU THÀNH ĐỒ THỊ TRI THỨC")
    print("=" * 65)
    print()

    print("📂 Đọc dữ liệu thô từ data/...")
    raw = load_raw_data()
    print(f"   → {len(raw)} file\n")

    if not raw:
        print("❌ Không có dữ liệu.")
        return

    print("🔍 Trích xuất 8 loại nút (NER)...")
    kg = {
        "University":         extract_university(raw),
        "Major":              extract_majors(raw),
        "SubjectCombination": extract_subject_combinations(raw),
        "AdmissionMethod":    extract_admission_methods(raw),
        "AdmissionCriteria":  extract_admission_criteria(raw),
        "MajorStatistic":     extract_major_statistics(raw),
        "ProgramType":        extract_program_types(),
    }
    kg["UniversityMajor"] = build_university_major(
        kg["University"], kg["Major"], kg["AdmissionCriteria"]
    )
    print()

    print("=" * 65)
    print("📊 THỐNG KÊ ĐỒ THỊ TRI THỨC")
    print("=" * 65)
    for nt, nodes in kg.items():
        print(f"   {nt:25s} : {len(nodes):5d} nút")
    print("=" * 65)
    print()

    print("📝 Sinh văn bản tự nhiên từ đồ thị (RE)...")
    texts_by_type = generate_all_texts(kg)
    print()

    print("💾 Lưu văn bản KG vào data/kg/...")
    save_kg_texts(texts_by_type)
    print()

    total = sum(len(v) for v in texts_by_type.values())
    print("=" * 65)
    print(f"✅ ĐÃ SINH {total} ĐOẠN VĂN BẢN KG")
    print(f"   Thư mục: {KG_DIR}")
    print("=" * 65)
    print()
    print("👉 Bước tiếp theo: python build_db.py")
    print()


if __name__ == "__main__":
    main()