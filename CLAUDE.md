# Trợ lý tư vấn ngành Khoa CNTT

Chatbot RAG (Streamlit) tư vấn tuyển sinh các ngành/chuyên ngành Khoa CNTT, Trường ĐH Điện lực.
Trả lời người dùng bằng tiếng Việt.

## Cấu trúc
- `app.py` — giao diện Streamlit (file lớn, ~1.500 dòng: chỉ đọc phần cần sửa)
- `rag_utils.py` — lõi RAG: lọc metadata, hybrid search (vector + BM25, RRF), rerank, dựng prompt, gọi LLM
- `external_llm.py` — gọi LLM cloud (OpenAI, Gemini, Claude, DeepSeek, Groq)
- `config.py` — mọi tham số (model, top-k, chunk, provider). Đổi tham số ở đây, không hard-code
- `build_db.py` — dựng kho vector `vector_db/` từ `data/*.txt` và `data/kg/*.txt`
- `build_kg.py` + `kg_schema.py` — sinh văn bản đồ thị tri thức vào `data/kg/`
- `score_converter.py` — quy đổi điểm học bạ sang thang THPT
- `data/` — dữ liệu nguồn. Các file `.txt` ở thư mục gốc là bản cũ, `build_db.py` không đọc chúng

## Lệnh
- Cài đặt: `pip install -r requirements.txt`
- Dựng đồ thị tri thức: `python build_kg.py`
- Dựng kho vector: `python build_db.py` (chạy lại sau mỗi lần sửa `data/` hoặc `data/kg/`)
- Chạy app: `streamlit run app.py`
- Kiểm tra (script in kết quả, không phải pytest):
  - `python test_score_converter.py`
  - `python test_kg.py`
  - `python test_search.py` (cần `vector_db/` đã dựng)

## Lưu ý
- `vector_db/` được commit vào repo: sau khi sửa dữ liệu phải dựng lại và commit cả kho mới.
- `EMBEDDING_MODEL` trong `config.py` là nguồn đúng; README có thể ghi model khác.
- API key đọc từ `st.secrets` rồi tới biến môi trường (`GEMINI_API_KEY`, ...). Không commit `.streamlit/secrets.toml` hay `.env`.
- Số liệu tuyển sinh (điểm chuẩn, học phí, mã ngành) phải lấy đúng năm trong `data/`, không suy ra từ năm khác.

## Cách làm việc
- Việc lớn: lập kế hoạch trước, ghi checklist vào `PLAN.md`.
- Khi phiên dài hoặc dừng giữa chừng: ghi trạng thái vào `HANDOFF.md` (đã xong, còn lại, quyết định, bước tiếp theo).
- Phiên mới: đọc `PLAN.md` / `HANDOFF.md` nếu có, không đọc lại toàn repo.
- Chạy script kiểm tra liên quan trước khi báo xong.
