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

from config import (
    LLM_TEMPERATURE,
    LLM_NUM_PREDICT,
    OPENAI_BASE_URL,
    DEEPSEEK_BASE_URL,
    GROQ_BASE_URL,
    PROVIDER_MODELS,
)


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

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_NUM_PREDICT,
            stream=True,
        )
    except Exception as e:
        raise RuntimeError(f"Lỗi gọi {model}: {e}")

    for chunk in stream:
        try:
            piece = chunk.choices[0].delta.content
        except (IndexError, AttributeError):
            piece = None
        if piece:
            yield piece


# ============================================================
# 2. GOOGLE GEMINI — TỰ ĐỘNG CHỌN SDK
# ============================================================

def stream_gemini(prompt: str, model: str, api_key: str):
    """
    ⚡ Tự động dùng SDK phù hợp:
    - Model gemini-2.0+ / 2.5 → google-genai (SDK mới)
    - Model gemini-1.5       → google-generativeai (SDK cũ)
    """
    if not api_key:
        raise RuntimeError("Thiếu API key Gemini.")

    # Model 2.0+ cần SDK mới
    if "2.0" in model or "2.5" in model:
        yield from _stream_gemini_new(prompt, model, api_key)
    else:
        yield from _stream_gemini_legacy(prompt, model, api_key)


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

    config = types.GenerateContentConfig(
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_NUM_PREDICT,
    )

    try:
        stream = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        raise RuntimeError(f"Lỗi gọi Gemini ({model}): {e}")

    for chunk in stream:
        try:
            text = chunk.text
        except Exception:
            text = None
        if text:
            yield text


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

    try:
        response = gm.generate_content(
            prompt,
            generation_config={
                "temperature": LLM_TEMPERATURE,
                "max_output_tokens": LLM_NUM_PREDICT,
            },
            stream=True,
        )
    except Exception as e:
        raise RuntimeError(f"Lỗi gọi Gemini ({model}): {e}")

    for chunk in response:
        try:
            text = chunk.text
        except Exception:
            text = None
        if text:
            yield text


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
        "gemini":   "gemini-1.5-flash",
        "claude":   "claude-3-5-sonnet-latest",
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