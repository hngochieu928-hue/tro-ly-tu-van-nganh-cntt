# ============================================================
# SOURCE_INFO.PY — TÊN HIỂN THỊ THÂN THIỆN CHO NGUỒN TÀI LIỆU
# ============================================================
# Ánh xạ tên file trong data/ sang tiêu đề, số hiệu văn bản và
# nhóm tài liệu để hiển thị cho người dùng thay vì tên file thô.
# ============================================================

import re

KIND_QUY_DINH = "Quy định"
KIND_TUYEN_SINH = "Tuyển sinh"
KIND_DIEM_CHUAN = "Điểm chuẩn"
KIND_CHUONG_TRINH = "Chương trình đào tạo"
KIND_GIOI_THIEU = "Giới thiệu"
KIND_TRI_THUC = "Cơ sở tri thức"

SOURCE_INFO = {
    "Chinh sach mien giam hoc phi ho tro chi phi hoc tap.txt": {
        "title": "Chính sách miễn, giảm học phí và hỗ trợ chi phí học tập",
        "ref": "QĐ 1089/QĐ-ĐHĐL · 12/05/2026",
        "kind": KIND_QUY_DINH,
    },
    "Muc thu hoc phi nam hoc 2026-2027.txt": {
        "title": "Mức thu học phí và các khoản thu khác năm học 2026–2027",
        "ref": "QĐ 1563/QĐ-ĐHĐL · 15/06/2026",
        "kind": KIND_QUY_DINH,
    },
    "Muc thu tien KTX co so 3 nam 2026-2027.txt": {
        "title": "Mức thu tiền ký túc xá Cơ sở 3, năm học 2026–2027",
        "ref": "QĐ 2518/QĐ-ĐHĐL · 17/08/2026",
        "kind": KIND_QUY_DINH,
    },
    "Quy dinh muc thu tien KTX.txt": {
        "title": "Quy định mức thu tiền lưu trú ký túc xá năm học 2026–2027",
        "ref": "QĐ 2183/QĐ-ĐHĐL · 16/07/2026",
        "kind": KIND_QUY_DINH,
    },
    "Quy che dao tao trinh do dai hoc.txt": {
        "title": "Quy chế đào tạo trình độ đại học",
        "ref": "QĐ 1835/QĐ-ĐHĐL · 22/10/2024",
        "kind": KIND_QUY_DINH,
    },
    "Quy dinh dong phuc le phuc sinh vien.txt": {
        "title": "Quy định về đồng phục, lễ phục của người học",
        "ref": "QĐ 423/QĐ-ĐHĐL · 06/02/2026",
        "kind": KIND_QUY_DINH,
    },
    "Quy dinh xet cap hoc bong sinh vien.txt": {
        "title": "Quy chế xét, cấp học bổng dành cho sinh viên",
        "ref": "QĐ 1018/QĐ-ĐHĐL · 24/04/2026",
        "kind": KIND_QUY_DINH,
    },
    "Quy trinh ky luat sinh vien.txt": {
        "title": "Quy trình công tác kỷ luật sinh viên",
        "ref": "QĐ 2022/QĐ-ĐHĐL · 07/07/2026",
        "kind": KIND_QUY_DINH,
    },
    "Tieu chi danh gia ket qua ren luyen sinh vien.txt": {
        "title": "Tiêu chí đánh giá kết quả rèn luyện sinh viên",
        "ref": "QĐ 543/QĐ-ĐHĐL · 12/03/2026",
        "kind": KIND_QUY_DINH,
    },
    "Quy_dinh_quy_doi_diem_hoc_ba_sang_THPT.txt": {
        "title": "Quy định quy đổi điểm học bạ sang điểm thi THPT",
        "ref": "Tuyển sinh năm 2026",
        "kind": KIND_QUY_DINH,
    },
    "Thong tin tuyen sinh nam 2024.txt": {
        "title": "Đề án tuyển sinh đại học năm 2024",
        "ref": "Trường Đại học Điện lực",
        "kind": KIND_TUYEN_SINH,
    },
    "Thong tin tuyen sinh nam 2025.txt": {
        "title": "Thông báo phương án tuyển sinh đại học chính quy năm 2025",
        "ref": "TB 654/TB-ĐHĐL · 20/03/2025",
        "kind": KIND_TUYEN_SINH,
    },
    "Thong tin tuyen sinh nam 2026.txt": {
        "title": "Thông tin tuyển sinh đại học chính quy năm 2026",
        "ref": "TB 466/TB-ĐHĐL · 16/01/2026",
        "kind": KIND_TUYEN_SINH,
    },
    "Diem_chuan_nam_2024.txt": {
        "title": "Điểm trúng tuyển đại học chính quy năm 2024",
        "ref": "Trường Đại học Điện lực",
        "kind": KIND_DIEM_CHUAN,
    },
    "Diem_chuan_nam_2025.txt": {
        "title": "Điểm trúng tuyển đại học chính quy năm 2025",
        "ref": "Trường Đại học Điện lực",
        "kind": KIND_DIEM_CHUAN,
    },
    "Diem_chuan_nam_2026.txt": {
        "title": "Thông báo điểm trúng tuyển đại học chính quy năm 2026",
        "ref": "Trường Đại học Điện lực",
        "kind": KIND_DIEM_CHUAN,
    },
    "chuong_trinh_khung_chuyen_nganh_an_ninh_mang.txt": {
        "title": "Chương trình khung — chuyên ngành An ninh mạng",
        "ref": "Khoa Công nghệ thông tin",
        "kind": KIND_CHUONG_TRINH,
    },
    "chuong_trinh_khung_chuyen_nganh_cong_nghe_phan_mem.txt": {
        "title": "Chương trình khung — chuyên ngành Công nghệ phần mềm",
        "ref": "Khoa Công nghệ thông tin",
        "kind": KIND_CHUONG_TRINH,
    },
    "chuong_trinh_khung_chuyen_nganh_he_thong_tmdt.txt": {
        "title": "Chương trình khung — chuyên ngành Hệ thống thương mại điện tử",
        "ref": "Khoa Công nghệ thông tin",
        "kind": KIND_CHUONG_TRINH,
    },
    "chuong_trinh_khung_nganh_khoa_hoc_du_lieu.txt": {
        "title": "Chương trình khung — ngành Khoa học dữ liệu",
        "ref": "Khoa Công nghệ thông tin",
        "kind": KIND_CHUONG_TRINH,
    },
    "gioi_thieu_nganh_chuyen_nganh.txt": {
        "title": "Giới thiệu ngành và chuyên ngành đào tạo",
        "ref": "Khoa Công nghệ thông tin",
        "kind": KIND_GIOI_THIEU,
    },
    "kg_university.txt": {
        "title": "Thông tin chung về Trường Đại học Điện lực",
        "ref": "Tổng hợp từ tài liệu tuyển sinh",
        "kind": KIND_TRI_THUC,
    },
    "kg_major.txt": {
        "title": "Danh mục ngành đào tạo và mã ngành",
        "ref": "Tổng hợp từ tài liệu tuyển sinh",
        "kind": KIND_TRI_THUC,
    },
    "kg_university_major.txt": {
        "title": "Ngành tuyển sinh theo từng năm",
        "ref": "Tổng hợp từ tài liệu tuyển sinh",
        "kind": KIND_TRI_THUC,
    },
    "kg_admission_criteria.txt": {
        "title": "Điểm chuẩn theo ngành và theo năm",
        "ref": "Tổng hợp từ thông báo điểm trúng tuyển",
        "kind": KIND_TRI_THUC,
    },
    "kg_subject_combination.txt": {
        "title": "Danh sách tổ hợp xét tuyển",
        "ref": "Tổng hợp từ thông tin tuyển sinh 2026",
        "kind": KIND_TRI_THUC,
    },
    "kg_admission_method.txt": {
        "title": "Các phương thức xét tuyển",
        "ref": "Tổng hợp từ thông tin tuyển sinh",
        "kind": KIND_TRI_THUC,
    },
    "kg_program_type.txt": {
        "title": "Các loại chương trình đào tạo",
        "ref": "Tổng hợp từ tài liệu tuyển sinh",
        "kind": KIND_TRI_THUC,
    },
}


def get_source_info(filename):
    info = SOURCE_INFO.get(filename)
    if info:
        return info

    name = re.sub(r"\.txt$", "", filename or "", flags=re.IGNORECASE)
    name = re.sub(r"[_\-]+", " ", name).strip()
    return {
        "title": name[:1].upper() + name[1:] if name else "Tài liệu",
        "ref": "",
        "kind": "Tài liệu",
    }


_STOPWORDS = {
    "là", "gì", "bao", "nhiêu", "của", "và", "có", "được", "không", "thế",
    "nào", "cho", "các", "những", "một", "khi", "em", "tôi", "mình", "hay",
    "với", "trong", "để", "này", "đó", "ở", "về", "như", "thì", "sao",
}


def _query_terms(query):
    words = re.findall(r"\w+", (query or "").lower())
    return [w for w in words if len(w) >= 2 and w not in _STOPWORDS]


def make_full_text(text, limit=1400):
    """Nội dung đầy đủ của đoạn để hiển thị khi mở rộng thẻ nguồn."""
    if not text:
        return ""

    text = re.sub(r"={3,}[^=\n]*={3,}", "", text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return text


def make_snippet(text, limit=240, query=None):
    """Rút gọn đoạn văn bản thành vài dòng xem trước; nếu có câu hỏi thì
    bắt đầu từ dòng khớp nhiều từ khóa nhất thay vì phần đầu đoạn."""
    if not text:
        return ""

    text = re.sub(r"={3,}[^=\n]*={3,}", " ", text)

    terms = _query_terms(query)
    if terms:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        scores = [
            sum(1 for t in terms if t in ln.lower()) for ln in lines
        ]
        if scores and max(scores) > 0:
            start = scores.index(max(scores))
            text = "\n".join(lines[start:])

    text = re.sub(r"\s*\|\s*", " · ", text)
    text = re.sub(r"\s+", " ", text).strip(" ·")

    if len(text) <= limit:
        return text

    cut = text[:limit]
    last_space = cut.rfind(" ")
    if last_space > limit * 0.6:
        cut = cut[:last_space]
    return cut.rstrip(" ,;·") + "…"
