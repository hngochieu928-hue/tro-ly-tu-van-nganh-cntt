# ============================================================
# KG_SCHEMA.PY — LƯỢC ĐỒ ĐỒ THỊ TRI THỨC
# ============================================================
# 11 loại nút + 10 loại quan hệ cho miền tuyển sinh và tư vấn ngành học.
# Đây là mô hình KHÁI NIỆM. Dữ liệu sinh văn bản → embedding →
# nạp vào ChromaDB phục vụ Hybrid RAG.
# ============================================================

NODE_TYPES = {
    "University":         {"properties": ["name", "code", "type", "address", "contact", "website"]},
    "Major":              {"properties": ["name", "code", "description", "career"]},
    "UniversityMajor":    {"properties": ["year", "tuition", "duration", "quota"]},
    "AdmissionCriteria":  {"properties": ["entranceScore", "year", "quota", "criteriaID"]},
    "SubjectCombination": {"properties": ["code", "subject1", "subject2", "subject3", "description"]},
    "AdmissionMethod":    {"properties": ["name", "formula", "description"]},
    "MajorStatistic":     {"properties": ["avgSalary", "employmentRate", "demand"]},
    "ProgramType":        {"properties": ["name", "description", "features"]},
    # --- mở rộng cho tư vấn ngành học (CNTT, Khoa học dữ liệu) ---
    "Specialization":     {"properties": ["name", "major_code", "major_name", "suitable_for", "knowledge"]},
    "Course":             {"properties": ["program", "major_code", "major_name", "semester", "courses"]},
    "Career":             {"properties": ["name", "major_code", "track", "skills", "courses", "workplaces"]},
}

RELATIONSHIP_TYPES = {
    "OFFERED_BY":           {"from": "UniversityMajor",   "to": "University"},
    "FOR_MAJOR":            {"from": "UniversityMajor",   "to": "Major"},
    "REQUIRES_COMBINATION": {"from": "AdmissionCriteria", "to": "SubjectCombination"},
    "APPLIES_TO":           {"from": "AdmissionCriteria", "to": "UniversityMajor"},
    "USES_METHOD":          {"from": "AdmissionCriteria", "to": "AdmissionMethod"},
    "HAS_PROGRAM_TYPE":     {"from": "UniversityMajor",   "to": "ProgramType"},
    "STATISTICS_FOR":       {"from": "MajorStatistic",    "to": "Major"},
    "SPECIALIZATION_OF":    {"from": "Specialization",     "to": "Major"},
    "COURSE_IN":            {"from": "Course",             "to": "Major"},
    "CAREER_FOR":           {"from": "Career",             "to": "Major"},
}

TEXT_TEMPLATES = {
    "University": (
        "{name} (mã trường {code}) là trường đại học {type} tại Việt Nam. "
        "Địa chỉ các trụ sở: {address}. Thông tin liên hệ tuyển sinh: {contact}. "
        "Website: {website}."
    ),
    "Major": (
        "Ngành {name} (mã ngành {code}): {description}. "
        "Cơ hội nghề nghiệp sau tốt nghiệp: {career}."
    ),
    "UniversityMajor": (
        "{university_name} tuyển sinh ngành {major_name} (mã ngành "
        "{major_code}) năm {year}. Chỉ tiêu: {quota}. Học phí: {tuition}. "
        "Thời gian đào tạo: {duration}."
    ),
    "AdmissionCriteria": (
        "Điểm chuẩn năm {year} ngành {major_name} tại Trường Đại học Điện lực: "
        "{entranceScore} điểm (thang điểm 30), chỉ tiêu {quota}. Phương thức "
        "xét tuyển: {method_names}."
    ),
    "SubjectCombination": (
        "Tổ hợp xét tuyển {code} gồm 3 môn: {subject1}, {subject2}, {subject3}."
    ),
    "AdmissionMethod": (
        "Phương thức xét tuyển: {name}. {description} Công thức tính điểm: {formula}."
    ),
    "MajorStatistic": (
        "Theo khảo sát năm {year}, sinh viên tốt nghiệp ngành {major_name} của "
        "Trường Đại học Điện lực có tỷ lệ có việc làm đạt {employmentRate}."
    ),
    "ProgramType": (
        "Loại chương trình đào tạo {name}: {description}. Đặc điểm: {features}."
    ),
    "Specialization": (
        "Chuyên ngành {name} là chuyên ngành thuộc ngành {major_name} tại Trường "
        "Đại học Điện lực. Mã ngành tương ứng: {major_code} (mã của ngành "
        "{major_name}). Phù hợp với sinh viên: {suitable_for}. "
        "Kiến thức được trang bị: {knowledge}."
    ),
    "Course": (
        "Chương trình khung {program_full}, học kỳ {semester} gồm các môn: {courses}."
    ),
    "Career": (
        "Vị trí nghề nghiệp {name}, thuộc định hướng {track}. Ngành đào tạo: "
        "{major_name}, mã ngành {major_code}. Kỹ năng cần có: {skills}. "
        "Môn học liên quan: {courses}. Nơi làm việc: {workplaces}."
    ),
}

def get_node_types():
    return list(NODE_TYPES.keys())

def get_relationship_types():
    return list(RELATIONSHIP_TYPES.keys())

def get_node_properties(node_type):
    return NODE_TYPES.get(node_type, {}).get("properties", [])

def validate_triplet(rel, src, dst):
    spec = RELATIONSHIP_TYPES.get(rel)
    if not spec:
        return False
    return spec["from"] == src and spec["to"] == dst