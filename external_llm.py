# ============================================================
# EXTERNAL_LLM.PY — STREAMING TỪ CÁC NHÀ CUNG CẤP ĐÁM MÂY
# ============================================================
# Hỗ trợ:
#   • OpenAI (ChatGPT)
#   • Google Gemini 1.5 (SDK cũ)
#   • Google Gemini 2.0+ (SDK mới — google-genai)
#   • Anthropic Claude
#   • DeepSeek
#   • Groq
# ============================================================

import time

from config import (
    LLM_TEMPERATURE,
    LLM_NUM_PREDICT,
    OPENAI_BASE_URL,
    DEEPSEEK_BASE_URL,
    GROQ_BASE_URL,
    PROVIDER_MODELS,
)


# ============================================================
# 0. RETRY CHO LỖI TẠM THỜI (503/429 — quá tải, rate limit)
# ============================================================

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 3

_TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded",
)


def _is_transient_error(exc) -> bool:
    text = str(exc)
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _stream_with_retry(make_stream, extract_text, error_prefix):
    """
    Mở stream bằng make_stream() và yield từng đoạn text qua extract_text(chunk).
    ⚡ QUAN TRỌNG: lỗi 503/429 thường chỉ lộ ra khi BẮT ĐẦU ĐỌC stream (lazy),
    không phải lúc gọi hàm khởi tạo — nên phải bọc retry quanh cả vòng lặp đọc,
    không chỉ quanh lệnh gọi API ban đầu.
    Chỉ tự thử lại khi CHƯA có đoạn text nào được yield ra ngoài (tránh lặp
    lại nội dung đã gửi cho người dùng nếu lỗi xảy ra giữa chừng).
    """
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        got_any = False
        try:
            for chunk in make_stream():
                text = extract_text(chunk)
                if text:
                    got_any = True
                    yield text
            return
        except Exception as e:
            last_error = e
            if not got_any and attempt < MAX_RETRIES and _is_transient_error(e):
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise RuntimeError(f"{error_prefix}: {e}")
    raise RuntimeError(f"{error_prefix}: {last_error}")


# ============================================================
# 1. OPENAI / DEEPSEEK / GROQ — DÙNG CHUNG SDK OpenAI
# ============================================================

def stream_openai_compatible(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str = None,
):
    """Stream cho mọi API tương thích OpenAI."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "Chưa cài openai. Chạy: pip install openai"
        )

    if not api_key:
        raise RuntimeError("Thiếu API key.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    def make_stream():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_NUM_PREDICT,
            stream=True,
        )

    def extract_text(chunk):
        try:
            return chunk.choices[0].delta.content
        except (IndexError, AttributeError):
            return None

    yield from _stream_with_retry(make_stream, extract_text, f"Lỗi gọi {model}")


# ============================================================
# 2. GOOGLE GEMINI — TỰ ĐỘNG CHỌN SDK
# ============================================================

def stream_gemini(prompt: str, model: str, api_key: str):
    """
    ⚡ Tự động dùng SDK phù hợp:
    - Model gemini-1.5       → google-generativeai (SDK cũ)
    - Model gemini-2.0 trở lên → google-genai (SDK mới)
    """
    if not api_key:
        raise RuntimeError("Thiếu API key Gemini.")

    # Chỉ model 1.5 dùng SDK cũ, mọi phiên bản mới hơn dùng SDK mới
    if "1.5" in model:
        yield from _stream_gemini_legacy(prompt, model, api_key)
    else:
        yield from _stream_gemini_new(prompt, model, api_key)


def _stream_gemini_new(prompt: str, model: str, api_key: str):
    """SDK mới: google-genai (Gemini 2.0+)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "Chưa cài google-genai. "
            "Chạy: pip install google-genai"
        )

    client = genai.Client(api_key=api_key)

    config_kwargs = {
        "temperature": LLM_TEMPERATURE,
        "max_output_tokens": LLM_NUM_PREDICT,
    }

    # ⚡ Tắt "thinking" (chain-of-thought ẩn): các model Gemini 2.5+/3.x mặc
    # định dành phần lớn max_output_tokens cho suy nghĩ nội bộ không hiển thị,
    # khiến câu trả lời thật bị cắt cụt gần như ngay khi vừa bắt đầu (đã xác
    # minh: thoughts_token_count chiếm ~860/900 token ngân sách). Tắt hẳn để
    # toàn bộ ngân sách token dành cho câu trả lời hiển thị.
    # Riêng model "lite" không hỗ trợ tham số này (trả lỗi 400 INVALID_ARGUMENT)
    # nên bỏ qua với các model có "lite" trong tên.
    if "lite" not in model:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=0
        )

    config = types.GenerateContentConfig(**config_kwargs)

    def make_stream():
        return client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=config,
        )

    def extract_text(chunk):
        try:
            return chunk.text
        except Exception:
            return None

    yield from _stream_with_retry(
        make_stream, extract_text, f"Lỗi gọi Gemini ({model})"
    )


def _stream_gemini_legacy(prompt: str, model: str, api_key: str):
    """SDK cũ: google-generativeai (Gemini 1.5)."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "Chưa cài google-generativeai. "
            "Chạy: pip install google-generativeai"
        )

    genai.configure(api_key=api_key)
    gm = genai.GenerativeModel(model)

    def make_stream():
        return gm.generate_content(
            prompt,
            generation_config={
                "temperature": LLM_TEMPERATURE,
                "max_output_tokens": LLM_NUM_PREDICT,
            },
            stream=True,
        )

    def extract_text(chunk):
        try:
            return chunk.text
        except Exception:
            return None

    yield from _stream_with_retry(
        make_stream, extract_text, f"Lỗi gọi Gemini ({model})"
    )


# ============================================================
# 3. ANTHROPIC CLAUDE
# ============================================================

def stream_claude(prompt: str, model: str, api_key: str):
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "Chưa cài anthropic. Chạy: pip install anthropic"
        )

    if not api_key:
        raise RuntimeError("Thiếu API key Anthropic.")

    client = anthropic.Anthropic(api_key=api_key)

    try:
        with client.messages.stream(
            model=model,
            max_tokens=LLM_NUM_PREDICT,
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
    except Exception as e:
        raise RuntimeError(f"Lỗi gọi Claude: {e}")


# ============================================================
# 4. DISPATCHER
# ============================================================

def stream_external(
    prompt: str,
    provider: str,
    api_key: str,
    model: str = None,
):
    """
    Điều phối tới đúng nhà cung cấp.
    `provider` ∈ {"openai", "gemini", "claude", "deepseek", "groq"}
    """
    provider = (provider or "").lower()

    default_models = {
        "openai":   "gpt-4o-mini",
        "gemini":   "gemini-flash-lite-latest",
        "claude":   "claude-sonnet-5",
        "deepseek": "deepseek-chat",
        "groq":     "llama-3.3-70b-versatile",
    }

    if model is None:
        model = default_models.get(provider)

    if provider == "openai":
        yield from stream_openai_compatible(
            prompt, model, api_key, base_url=OPENAI_BASE_URL
        )

    elif provider == "deepseek":
        yield from stream_openai_compatible(
            prompt, model, api_key, base_url=DEEPSEEK_BASE_URL
        )

    elif provider == "groq":
        yield from stream_openai_compatible(
            prompt, model, api_key, base_url=GROQ_BASE_URL
        )

    elif provider == "gemini":
        yield from stream_gemini(prompt, model, api_key)

    elif provider == "claude":
        yield from stream_claude(prompt, model, api_key)

    else:
        raise ValueError(f"Provider không hỗ trợ: {provider}")