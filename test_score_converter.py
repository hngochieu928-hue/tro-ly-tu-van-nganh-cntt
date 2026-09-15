# ============================================================
# TEST_SCORE_CONVERTER.PY
# ============================================================
from score_converter import (
    convert_score, convert_with_priority,
    explain_formula, batch_convert,
    DEFAULT_PERCENTILE_BANDS,
)


def section(title):
    print("\n" + "=" * 68)
    print(f"   {title}")
    print("=" * 68)


def main():
    print()
    print("=" * 68)
    print("   TEST CÔNG THỨC QUY ĐỔI ĐIỂM HỌC BẠ → THPT")
    print("=" * 68)

    m, n = DEFAULT_PERCENTILE_BANDS["hoc_ba"]
    a, b = DEFAULT_PERCENTILE_BANDS["thpt"]

    section("1. MỐC PHÂN VỊ MẶC ĐỊNH")
    print(f"   Học bạ : [m, n] = [{m}, {n}]")
    print(f"   THPT   : [a, b] = [{a}, {b}]")

    section("2. VÍ DỤ CHI TIẾT (x = 24)")
    print(explain_formula(24.0))

    section("3. BẢNG QUY ĐỔI ĐẦY ĐỦ")
    print(f"   {'Học bạ (x)':>12s}  →  {'THPT (y)':>12s}")
    print("   " + "-" * 32)
    for x in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        y = convert_score(x)
        print(f"   {x:>12.2f}  →  {y:>12.2f}")

    section("4. TEST BIÊN (m, n)")
    for x in [m, n]:
        y = convert_score(x)
        expected = a if x == m else b
        mark = "✅" if abs(y - expected) < 0.01 else "❌"
        print(f"   {mark} x = {x:>5.2f}  →  y = {y:>5.2f}  (kỳ vọng: {expected})")

    section("5. TEST ĐIỂM ƯU TIÊN")
    print("   Trường hợp 1: Điểm sau quy đổi < 22.5 (KHÔNG giảm ưu tiên)")
    r1 = convert_with_priority(20.0, diem_uu_tien=2.0)
    for k, v in r1.items():
        print(f"      {k:22s}: {v}")

    print("\n   Trường hợp 2: Điểm sau quy đổi ≥ 22.5 (CÓ giảm ưu tiên)")
    r2 = convert_with_priority(26.0, diem_uu_tien=2.0)
    for k, v in r2.items():
        print(f"      {k:22s}: {v}")

    section("6. TEST BATCH")
    for r in batch_convert([17, 18, 22, 26, 30, 31]):
        if r["error"]:
            print(f"   x = {r['x']:>4}  →  ❌ {r['error']}")
        else:
            print(f"   x = {r['x']:>4}  →  y = {r['y']:>6.2f}")

    section("7. TEST BẮT LỖI")
    test_cases = [
        ("x < m",  lambda: convert_score(17.0)),
        ("x > n",  lambda: convert_score(31.0)),
        ("n <= m", lambda: convert_score(24.0, m=30, n=18)),
        ("b <= a", lambda: convert_score(24.0, a=30, b=15)),
    ]
    for label, fn in test_cases:
        try:
            fn()
            print(f"   ❌ {label:10s}: KHÔNG bắt được lỗi")
        except ValueError as e:
            print(f"   ✅ {label:10s}: {e}")

    print()
    print("=" * 68)
    print("✅ HOÀN THÀNH TEST CÔNG THỨC QUY ĐỔI ĐIỂM")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()