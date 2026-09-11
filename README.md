# Trợ lý tư vấn ngành Khoa CNTT

Chatbot RAG (Retrieval-Augmented Generation) tư vấn tuyển sinh về các ngành/chuyên ngành của Khoa Công nghệ thông tin: Công nghệ phần mềm, Quản trị và An ninh mạng, Hệ thống Thương mại điện tử, Khoa học dữ liệu.

- **Tìm kiếm**: Hybrid search (vector + BM25) + reranker (`BAAI/bge-reranker-v2-m3`)
- **Embedding**: `BAAI/bge-m3` chạy local qua `sentence-transformers` (không phụ thuộc dịch vụ ngoài)
- **LLM**: hỗ trợ nhiều nhà cung cấp — Ollama (local), OpenAI, Google Gemini, Anthropic Claude, DeepSeek, Groq
- **Giao diện**: Streamlit

## Chạy ở máy local

```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
venv\Scripts\python build_db.py     # build vector DB từ data/*.txt
venv\Scripts\streamlit run app.py
```

Muốn dùng LLM chạy local (không tốn phí API) thì cài thêm [Ollama](https://ollama.com), tải model:

```bash
ollama pull qwen2.5:3b
```

rồi chọn "🖥️ Ollama local" trong sidebar của app.

## Deploy public (Hugging Face Spaces — miễn phí)

Vì Ollama không chạy được trên hosting miễn phí, bản deploy public dùng **Google Gemini** làm LLM mặc định (embedding + reranker vẫn chạy local, miễn phí, không cần API key).

1. Lấy API key miễn phí tại https://aistudio.google.com/apikey
2. Vào https://huggingface.co/new-space → đăng nhập bằng GitHub
   - Space SDK: **Streamlit**
   - Import từ repo GitHub này, hoặc push code lên Space bằng git
3. Vào **Settings → Variables and secrets** của Space, thêm secret:
   - `GEMINI_API_KEY` = API key vừa lấy ở bước 1
4. Space sẽ tự cài `requirements.txt` và chạy `app.py`. Lần khởi động đầu sẽ hơi lâu vì phải tải model `bge-m3` + `bge-reranker-v2-m3` (~4GB tổng).

Sau khi deploy xong, ai có link Space đều dùng được ngay, không cần tự nhập API key.

## Cấu trúc

- `app.py` — giao diện Streamlit
- `rag.py` — bản dòng lệnh (CLI), mặc định dùng Ollama local
- `rag_utils.py` — logic RAG cốt lõi (embedding, hybrid search, rerank, gọi LLM)
- `build_db.py` — build vector DB từ `data/*.txt`
- `external_llm.py` — wrapper gọi các LLM cloud
- `config.py` — cấu hình tập trung
- `data/` — dữ liệu nguồn (chương trình đào tạo, giới thiệu ngành)
