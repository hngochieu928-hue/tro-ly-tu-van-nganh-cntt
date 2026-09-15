# ============================================================
# KG_SCHEMA.PY — LƯỢC ĐỒ ĐỒ THỊ TRI THỨC
# ============================================================
# 8 loại nút + 7 loại quan hệ cho miền tuyển sinh.
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
}

RELATIONSHIP_TYPES = {
    "OFFERED_BY":           {"from": "UniversityMajor",   "to": "University"},
    "FOR_MAJOR":            {"from": "UniversityMajor",   "to": "Major"},
    "REQUIRES_COMBINATION": {"from": "AdmissionCriteria", "to": "SubjectCombination"},
    "APPLIES_TO":           {"from": "AdmissionCriteria", "to": "UniversityMajor"},
    "USES_METHOD":          {"from": "AdmissionCriteria", "to": "AdmissionMethod"},
    "HAS_PROGRAM_TYPE":     {"from": "UniversityMajor",   "to": "ProgramType"},
    "STATISTICS_FOR":       {"from": "MajorStatistic",    "to": "Major"},
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
        "Trường {university_name} tuyển sinh ngành {major_name} (mã ngành "
        "{major_code}) năm {year} với chỉ tiêu {quota}. Học phí: {tuition}. "
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