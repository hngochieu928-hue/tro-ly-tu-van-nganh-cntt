# ============================================================
# SCORE_CONVERTER.PY — QUY ĐỔI ĐIỂM HỌC BẠ → THPT
# ============================================================
# Công thức nội suy tuyến tính chính thức của EPU:
#       y = a + (x - m) / (n - m) × (b - a)
# ============================================================

DEFAULT_PERCENTILE_BANDS = {
    "hoc_ba": (18.0, 30.0),
    "thpt":   (15.0, 30.0),
}


def convert_score(x, m=None, n=None, a=None, b=None):
    """Quy đổi điểm học bạ x sang điểm THPT tương đương y."""
    if m is None or n is None:
        m, n = DEFAULT_PERCENTILE_BANDS["hoc_ba"]
    if a is None or b is None:
        a, b = DEFAULT_PERCENTILE_BANDS["thpt"]

    if n <= m:
        raise ValueError(f"Khoảng [m, n] không hợp lệ: n={n} <= m={m}")
    if b <= a:
        raise ValueError(f"Khoảng [a, b] không hợp lệ: b={b} <= a={a}")
    if not (m <= x <= n):
        raise ValueError(f"Điểm x={x} nằm ngoài khoảng học bạ [{m}, {n}]")

    y = a + (x - m) / (n - m) * (b - a)
    y = max(a, min(b, y))
    return round(y, 2)


def convert_with_priority(x, diem_uu_tien=0.0, m=None, n=None, a=None, b=None):
    """Quy đổi điểm + cộng điểm ưu tiên + áp dụng quy tắc giảm điểm ưu tiên."""
    y = convert_score(x, m, n, a, b)

    if y >= 22.5 and diem_uu_tien > 0:
        diem_ut_thuc = round((30 - y) / 7.5 * diem_uu_tien, 2)
    else:
        diem_ut_thuc = diem_uu_tien

    return {
        "x_hoc_ba":           x,
        "y_thpt_tuong_duong": y,
        "diem_uu_tien_goc":   diem_uu_tien,
        "diem_uu_tien_thuc":  diem_ut_thuc,
        "tong_diem_xet":      round(y + diem_ut_thuc, 2),
    }


def explain_formula(x, m=None, n=None, a=None, b=None):
    """Trả về giải thích chi tiết từng bước tính toán."""
    if m is None or n is None:
        m, n = DEFAULT_PERCENTILE_BANDS["hoc_ba"]
    if a is None or b is None:
        a, b = DEFAULT_PERCENTILE_BANDS["thpt"]

    y = convert_score(x, m, n, a, b)

    return (
        f"Quy đổi điểm học bạ x = {x} sang điểm THPT tương đương:\n\n"
        f"    Công thức: y = a + (x - m) / (n - m) × (b - a)\n\n"
        f"    Với:\n"
        f"        [m, n] = [{m}, {n}]  (khoảng điểm học bạ)\n"
        f"        [a, b] = [{a}, {b}]  (khoảng điểm THPT)\n\n"
        f"    Thay số:\n"
        f"        y = {a} + ({x} - {m}) / ({n} - {m}) × ({b} - {a})\n"
        f"          = {a} + {round(x - m, 2)} / {round(n - m, 2)} × {round(b - a, 2)}\n"
        f"          = {y}\n\n"
        f"    Vậy: Điểm học bạ {x} → Điểm THPT tương đương {y}"
    )


def batch_convert(scores, **kwargs):
    """Quy đổi nhiều điểm cùng lúc."""
    results = []
    for x in scores:
        try:
            y = convert_score(x, **kwargs)
            results.append({"x": x, "y": y, "error": None})
        except ValueError as e:
            results.append({"x": x, "y": None, "error": str(e)})
    return results


if __name__ == "__main__":
    print(explain_formula(24.0))
    print()
    for x in [18, 20, 22, 24, 26, 28, 30]:
        print(f"{x:>5.2f} → {convert_score(x):>5.2f}")