# ============================================================
# APP.PY — TRỢ LÝ TƯ VẤN NGÀNH KHOA CNTT
# Model: Multi-LLM (Ollama / OpenAI / Gemini / Claude / DeepSeek / Groq)
# Hybrid+Rerank | Streaming | Sidebar
# ============================================================

import os
import time
import base64
import uuid
from datetime import datetime

import streamlit as st

from config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    TOP_K,
    FINAL_TOP_K,
    DEFAULT_PROVIDER,
    PROVIDER_MODELS,
    PROVIDER_LABELS,
    PROVIDER_KEY_LABELS,
)

from rag_utils import (
    ask_ai,
    get_chroma_collection,
    prepare_rag_prompt,
    generate_answer_stream,
    warm_up,
    clean_answer,
)


# ============================================================
# 1. CẤU HÌNH TRANG
# ============================================================

st.set_page_config(
    page_title="Trợ lý tư vấn ngành Khoa CNTT",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# 1b. NẠP API KEYS TỪ SECRETS / ENV
# ============================================================

def _pick_secret(name: str) -> str:
    """Ưu tiên: st.secrets → biến môi trường → rỗng."""
    try:
        val = st.secrets.get(name)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(name, "")


def _init_api_keys():
    """Khởi tạo session_state.api_keys từ secrets / env."""
    if "api_keys" in st.session_state:
        return
    st.session_state.api_keys = {
        "openai":   _pick_secret("OPENAI_API_KEY"),
        "gemini":   _pick_secret("GEMINI_API_KEY"),
        "claude":   _pick_secret("ANTHROPIC_API_KEY"),
        "deepseek": _pick_secret("DEEPSEEK_API_KEY"),
        "groq":     _pick_secret("GROQ_API_KEY"),
    }


_init_api_keys()


# ============================================================
# 2. TỰ ĐỘNG TÌM & ĐỌC LOGO
# ============================================================

LOGO_CANDIDATES = [
    "logo.jpg",
    "logo.jpeg",
    "logo.png",
    "logo.webp",
]

LOGO_PATH = None
for candidate in LOGO_CANDIDATES:
    if os.path.exists(candidate):
        LOGO_PATH = candidate
        break

LOGO_B64 = None
if LOGO_PATH:
    try:
        with open(LOGO_PATH, "rb") as f:
            LOGO_B64 = base64.b64encode(f.read()).decode()
    except Exception:
        LOGO_B64 = None

if LOGO_B64:
    LOGO_SIDEBAR = (
        f'<img src="data:image/jpeg;base64,{LOGO_B64}" '
        f'class="custom-logo" alt="Logo Khoa CNTT">'
    )
    LOGO_HERO = (
        f'<img src="data:image/jpeg;base64,{LOGO_B64}" '
        f'class="hero-icon" alt="Logo Khoa CNTT">'
    )
    ASSISTANT_AVATAR = LOGO_PATH
else:
    LOGO_SIDEBAR = '<div class="brand-logo">CNS</div>'
    LOGO_HERO = '<div class="brand-logo hero-icon-text">CNS</div>'
    ASSISTANT_AVATAR = "🎓"


# ============================================================
# 3. CSS  (giữ nguyên như bản gốc)
# ============================================================

st.markdown(
    """
<style>
/* ═══════════════════════════════════════════
   ẨN CHROME MẶC ĐỊNH
   ═══════════════════════════════════════════ */
#MainMenu, footer { visibility: hidden; }

header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem !important;
}

[data-testid="stSidebarCollapsedControl"] button {
    background: #ffffff !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 8px !important;
    padding: 6px 8px !important;
}
[data-testid="stSidebarCollapsedControl"] svg {
    fill: #17191d !important;
}

/* ═══════════════════════════════════════════
   NỀN & FONT
   ═══════════════════════════════════════════ */
html, body, .stApp {
    background: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif;
}

[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    max-width: 880px;
    padding-top: 2.2rem;
    padding-bottom: 6rem;
}

/* ═══════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #f7f7f8;
    border-right: 1px solid #ececf1;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 4px 18px 4px;
}
.custom-logo {
    width: 42px; height: 42px;
    object-fit: contain;
    border-radius: 8px;
    background: #ffffff;
}
.brand-logo {
    width: 38px; height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #315bea 0%, #6c8bff 100%);
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
}
.brand-text { line-height: 1.15; }
.brand-text .t1 {
    font-size: 14px; font-weight: 700; color: #17191d;
}
.brand-text .t2 {
    font-size: 11.5px; color: #8a90a0;
}

.side-heading {
    font-size: 11.5px;
    color: #8e8ea0;
    font-weight: 600;
    margin: 18px 0 8px 6px;
}

/* ═══ Nút PRIMARY ═══ */
[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
    width: 100%;
    background: #ffffff !important;
    color: #17191d !important;
    border: 1px solid #e5e5e5 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.04) !important;
    text-align: left !important;
    justify-content: flex-start !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] p {
    color: #17191d !important;
    text-align: left !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* ═══ Nút SECONDARY ═══ */
[data-testid="stSidebar"] div[data-testid="stButton"] > button:not([kind="primary"]) {
    width: 100%;
    background: transparent !important;
    color: #17191d !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 400 !important;
    font-size: 13.5px !important;
    padding: 8px 10px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: none !important;
    min-height: 34px !important;
    white-space: nowrap !important;
    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.3;
    margin-bottom: 1px;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:not([kind="primary"]):hover {
    background: #ececf1 !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:not([kind="primary"]) p {
    text-align: left !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    margin: 0 !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: inherit !important;
}

/* ═══ Nút menu (⋯) ═══ */
[data-testid="stSidebar"] div[data-testid="stPopover"] > button {
    position: relative !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    min-height: 34px !important;
    height: 34px !important;
    width: 100% !important;
    box-shadow: none !important;
    border-radius: 8px !important;
    opacity: 0.6;
    overflow: hidden !important;
}
[data-testid="stSidebar"] div[data-testid="stPopover"] > button > *,
[data-testid="stSidebar"] div[data-testid="stPopover"] > button svg {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
}
[data-testid="stSidebar"] div[data-testid="stPopover"] > button::after {
    content: "..." !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #8e8ea0 !important;
    font-size: 12px !important;
    font-weight: 900 !important;
    letter-spacing: 1px !important;
    line-height: 1 !important;
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    pointer-events: none !important;
}
[data-testid="stSidebar"] div[data-testid="stPopover"] > button:hover {
    background: #d9d9e0 !important;
    opacity: 1;
}
[data-testid="stSidebar"] div[data-testid="stPopover"] > button:hover::after {
    color: #17191d !important;
}
[data-testid="stPopoverBody"] {
    border-radius: 12px !important;
    padding: 6px !important;
    box-shadow: 0 10px 30px rgba(16,24,40,.18) !important;
    border: 1px solid #e5e5e5 !important;
    background: #ffffff !important;
    min-width: 160px !important;
}
[data-testid="stPopoverBody"] button {
    background: transparent !important;
    color: #17191d !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    margin-bottom: 1px !important;
    box-shadow: none !important;
    min-height: auto !important;
}
[data-testid="stPopoverBody"] button:hover {
    background: #f0f0f5 !important;
}
[data-testid="stPopoverBody"] button p {
    text-align: left !important;
    color: inherit !important;
    font-size: 13.5px !important;
    margin: 0 !important;
}

/* ═══════════════════════════════════════════
   HERO
   ═══════════════════════════════════════════ */
.hero {
    text-align: center;
    margin: 18px 0 34px 0;
    animation: fadeUp .55s ease;
}
.hero-icon {
    width: 120px; height: 120px;
    margin: 0 auto 22px auto;
    object-fit: contain;
    background: #ffffff;
    border-radius: 20px;
    padding: 6px;
}
.hero-icon-text {
    width: 90px; height: 90px;
    margin: 0 auto 22px auto;
    font-size: 24px;
    border-radius: 22px;
}
.hero h1 {
    color: #14161c;
    font-size: 28px;
    line-height: 1.35;
    margin: 0 0 12px 0;
    font-weight: 800;
    letter-spacing: -.5px;
}
.hero h1 .accent {
    background: linear-gradient(135deg, #315bea 0%, #7b96ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #7a8194;
    font-size: 15px;
    line-height: 1.65;
    max-width: 560px;
    margin: 0 auto;
}

/* ═══════════════════════════════════════════
   CHAT MESSAGE
   ═══════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
    margin: 6px 0 !important;
    animation: fadeUp .35s ease;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] {
    background: #f1f3f5;
    padding: 12px 18px;
    border-radius: 20px 20px 6px 20px;
    color: #17191d;
    font-size: 15px;
    line-height: 1.6;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] strong {
    color: #17191d !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
[data-testid="stChatMessageContent"] {
    background: #ffffff;
    color: #24272c;
    font-size: 15px;
    line-height: 1.75;
    padding: 16px 20px;
    border-radius: 4px 20px 20px 20px;
    border: 1px solid #eef0f5;
    box-shadow: 0 4px 14px rgba(16,24,40,.05);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    margin: 0 0 12px 0;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child {
    margin-bottom: 0;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
    color: #14161c;
    font-weight: 700;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
    color: #14161c;
    margin: 18px 0 10px 0;
    font-weight: 700;
    line-height: 1.35;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ol {
    margin: 0 0 12px 0;
    padding-left: 22px;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    margin-bottom: 5px;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
    background: #f1f3f8;
    padding: 2px 6px;
    border-radius: 5px;
    font-size: 13px;
    color: #315bea;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {
    border-collapse: collapse;
    width: 100%;
    margin: 14px 0;
    font-size: 14px;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
    border: 1px solid #e8ebf2;
    padding: 9px 12px;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {
    background: #f5f7fb;
    font-weight: 700;
}
[data-testid="stChatMessageAvatarUser"] {
    background: #eef3ff !important;
    color: #315bea !important;
    border: none !important;
    font-weight: 700 !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: #ffffff !important;
    border: 1px solid #e8ebf2 !important;
    overflow: hidden !important;
}
[data-testid="stChatMessageAvatarAssistant"] img {
    object-fit: contain !important;
    padding: 2px;
    background: #ffffff;
}

/* ═══════════════════════════════════════════
   EXPANDER
   ═══════════════════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid #e8ebf2 !important;
    border-radius: 14px !important;
    margin-top: 12px !important;
    background: #ffffff !important;
}
[data-testid="stExpander"]:hover {
    border-color: #cdd8ff !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important;
    color: #454a51 !important;
    font-weight: 600 !important;
    padding: 11px 16px !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    font-size: 13px;
    margin: 4px 0;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] code {
    background: #eef3ff;
    color: #315bea;
    padding: 1px 6px;
    border-radius: 5px;
}

/* ═══════════════════════════════════════════
   CHAT INPUT
   ═══════════════════════════════════════════ */
[data-testid="stChatInput"] {
    border-radius: 26px !important;
    border: 1px solid #dfe3ec !important;
    background: #ffffff !important;
    box-shadow: 0 6px 20px rgba(16,24,40,.08) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #315bea !important;
    box-shadow: 0 8px 26px rgba(49,91,234,.18) !important;
}
[data-testid="stChatInput"] textarea {
    font-size: 15px !important;
    padding: 12px 4px !important;
}
.stSpinner > div { border-top-color: #315bea !important; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 4. SESSION STATE
# ============================================================

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "current_id" not in st.session_state:
    st.session_state.current_id = None

if "collection" not in st.session_state:
    st.session_state.collection = None

if "ready" not in st.session_state:
    st.session_state.ready = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "renaming_id" not in st.session_state:
    st.session_state.renaming_id = None

# --- Multi-LLM ---
if "provider" not in st.session_state:
    st.session_state.provider = DEFAULT_PROVIDER

if "cloud_models" not in st.session_state:
    st.session_state.cloud_models = {
        p: PROVIDER_MODELS[p][0]
        for p in PROVIDER_MODELS
        if p != "ollama"
    }

if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = LLM_MODEL


# ============================================================
# 5. QUẢN LÝ CONVERSATION
# ============================================================

def create_new_conversation() -> str:
    conv_id = str(uuid.uuid4())
    st.session_state.conversations.append({
        "id": conv_id,
        "title": "Cuộc trò chuyện mới",
        "messages": [],
        "pinned": False,
        "created_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
    })
    st.session_state.current_id = conv_id
    return conv_id


def get_current_conversation():
    if not st.session_state.current_id:
        return None
    for conv in st.session_state.conversations:
        if conv["id"] == st.session_state.current_id:
            return conv
    return None


def update_conversation_title(conv, question: str):
    if conv["title"] == "Cuộc trò chuyện mới" and question:
        title = question.strip()
        if len(title) > 40:
            title = title[:40] + "..."
        conv["title"] = title


def delete_conversation(conv_id: str):
    st.session_state.conversations = [
        c for c in st.session_state.conversations
        if c["id"] != conv_id
    ]
    if st.session_state.current_id == conv_id:
        if st.session_state.conversations:
            st.session_state.current_id = (
                st.session_state.conversations[-1]["id"]
            )
        else:
            st.session_state.current_id = None


def toggle_pin(conv_id: str):
    for conv in st.session_state.conversations:
        if conv["id"] == conv_id:
            conv["pinned"] = not conv.get("pinned", False)
            break


def rename_conversation(conv_id: str, new_title: str):
    new_title = new_title.strip()
    if not new_title:
        return
    for conv in st.session_state.conversations:
        if conv["id"] == conv_id:
            conv["title"] = new_title[:60]
            break


# ============================================================
# 6. KẾT NỐI DATABASE + WARM-UP
# ============================================================

if not st.session_state.ready:
    try:
        st.session_state.collection = get_chroma_collection()
        st.session_state.ready = True
        st.session_state.pop("db_error", None)
    except Exception as e:
        st.session_state.db_error = str(e)

if "warmed_up" not in st.session_state and st.session_state.ready:
    with st.spinner(
        "⚙️ Đang khởi động RAG (Hybrid + Rerank)..."
    ):
        st.session_state.warmed_up = warm_up(
            provider=st.session_state.provider
        )


# ============================================================
# 7. HIỂN THỊ NGUỒN TÀI LIỆU
# ============================================================

def render_sources(metadatas):
    if not metadatas:
        return

    seen, unique = set(), []
    for meta in metadatas:
        key = (meta.get("source", "?"), meta.get("chunk", "?"))
        if key not in seen:
            seen.add(key)
            unique.append(meta)

    with st.expander(
        f"📚 Nguồn tài liệu tham khảo ({len(unique)})",
        expanded=False,
    ):
        for meta in unique:
            source = meta.get("source", "Không xác định")
            chunk = meta.get("chunk", "?")
            score = meta.get("score")

            if score is not None:
                st.markdown(
                    f"- `{source}` — chunk **{chunk}** "
                    f"(score: `{score:.4f}`)"
                )
            else:
                st.markdown(f"- `{source}` — chunk **{chunk}**")


# ============================================================
# 8. XỬ LÝ CÂU HỎI
# ============================================================

def process_question(question: str) -> None:
    question = question.strip()
    if not question:
        return

    conv = get_current_conversation()
    if conv is None:
        create_new_conversation()
        conv = get_current_conversation()

    if not st.session_state.ready:
        st.error("❌ Không thể kết nối tới ChromaDB.")
        if "db_error" in st.session_state:
            with st.expander("Chi tiết lỗi"):
                st.code(st.session_state.db_error)
        return

    conv["messages"].append({"role": "user", "content": question})
    update_conversation_title(conv, question)
    history = conv["messages"][:-1][-6:]

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    start = time.perf_counter()
    answer = ""
    metadatas = []

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):

        with st.spinner("🔎 Đang tra cứu kho dữ liệu (Hybrid + Rerank)..."):
            try:
                prompt, metadatas = prepare_rag_prompt(
                    question,
                    st.session_state.collection,
                    FINAL_TOP_K,
                    history,
                )
            except Exception as e:
                st.error(f"Lỗi truy xuất dữ liệu: {e}")
                return

        if prompt is None:
            answer = (
                "Hiện tại tôi chưa tìm thấy thông tin phù hợp "
                "trong kho dữ liệu."
            )
            st.markdown(answer)
        else:
            placeholder = st.empty()
            try:
                provider = st.session_state.provider
                api_key  = st.session_state.api_keys.get(provider, "")

                if provider == "ollama":
                    model = st.session_state.ollama_model
                else:
                    model = st.session_state.cloud_models.get(provider)

                for chunk in generate_answer_stream(
                    prompt,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                ):
                    answer += chunk
                    placeholder.markdown(answer + "▌")
                answer = clean_answer(answer)
                placeholder.markdown(answer)
            except Exception as e:
                answer = f"Xin lỗi, hệ thống gặp lỗi.\n\n`{e}`"
                placeholder.markdown(answer)

        render_sources(metadatas)

        elapsed = time.perf_counter() - start
        st.caption(f"⏱️ Phản hồi trong {elapsed:.2f} giây")

    conv["messages"].append({
        "role": "assistant",
        "content": answer,
        "metadatas": metadatas,
        "time": elapsed,
    })


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    # ----- Brand -----
    st.markdown(
        f"""
        <div class="brand">
            {LOGO_SIDEBAR}
            <div class="brand-text">
                <div class="t1">Khoa CNTT</div>
                <div class="t2">Trợ lý tư vấn tuyển sinh</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Nút tạo cuộc trò chuyện mới -----
    if st.button(
        "＋  Cuộc trò chuyện mới",
        use_container_width=True,
        type="primary",
    ):
        create_new_conversation()
        st.session_state.pending_question = None
        st.rerun()

    # ----- 🤖 MODEL SELECTOR -----
    st.markdown(
        '<div class="side-heading">🤖 Mô hình AI</div>',
        unsafe_allow_html=True,
    )

    provider = st.selectbox(
        "Nhà cung cấp",
        options=list(PROVIDER_LABELS.keys()),
        format_func=lambda k: PROVIDER_LABELS[k],
        index=list(PROVIDER_LABELS.keys()).index(
            st.session_state.provider
        ),
        label_visibility="collapsed",
        key="provider_select_widget",
    )

    if provider != st.session_state.provider:
        st.session_state.provider = provider
        with st.spinner("⚙️ Đang chuyển mô hình..."):
            st.session_state.warmed_up = warm_up(provider=provider)
        st.rerun()

    # ----- Chọn model cụ thể -----
    if provider == "ollama":
        st.session_state.ollama_model = st.selectbox(
            "Model Ollama",
            options=PROVIDER_MODELS["ollama"],
            index=PROVIDER_MODELS["ollama"].index(
                st.session_state.ollama_model
            ) if st.session_state.ollama_model in PROVIDER_MODELS["ollama"]
            else 0,
            label_visibility="collapsed",
            key="ollama_model_select",
        )
        st.caption(f"🖥️ Đang dùng model local: `{st.session_state.ollama_model}`")

    else:
        choices = PROVIDER_MODELS[provider]
        current = st.session_state.cloud_models.get(provider, choices[0])
        idx = choices.index(current) if current in choices else 0

        chosen = st.selectbox(
            "Model cloud",
            options=choices,
            index=idx,
            label_visibility="collapsed",
            key=f"cloud_model_{provider}",
        )
        st.session_state.cloud_models[provider] = chosen

        # ----- API key -----
        current_key = st.session_state.api_keys.get(provider, "")
        key_widget = st.text_input(
            f"🔑 {PROVIDER_KEY_LABELS.get(provider, 'API key')}",
            value=current_key,
            type="password",
            placeholder="sk-... / AIza...",
            label_visibility="collapsed",
            key=f"apikey_{provider}",
        )
        st.session_state.api_keys[provider] = key_widget

        if not key_widget:
            st.warning(
                f"⚠️ Chưa có API key cho **{provider}**. "
                f"Hãy nhập key hoặc khai báo trong "
                f"`.streamlit/secrets.toml`.",
                icon="🔒",
            )

    # ----- Danh sách hội thoại -----
    st.markdown(
        '<div class="side-heading">Gần đây</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.conversations:
        st.markdown(
            "<p style='font-size: 12.5px; color: #a0a5ad; "
            "padding-left: 6px;'>Chưa có cuộc trò chuyện nào.</p>",
            unsafe_allow_html=True,
        )
    else:
        pinned   = [c for c in st.session_state.conversations if c.get("pinned")]
        unpinned = [c for c in st.session_state.conversations if not c.get("pinned")]
        ordered  = list(reversed(pinned)) + list(reversed(unpinned))

        for conv in ordered:
            is_pinned = conv.get("pinned", False)

            col_title, col_menu = st.columns([9, 1])

            with col_title:
                title = conv["title"]
                if is_pinned:
                    title = "📌 " + title

                if st.button(
                    title,
                    key=f"conv_{conv['id']}",
                    use_container_width=True,
                ):
                    st.session_state.current_id = conv["id"]
                    st.rerun()

            with col_menu:
                unique_label = f"m-{conv['id'][:8]}"
                with st.popover(unique_label, use_container_width=True):

                    pin_text = "Bỏ ghim" if is_pinned else "Ghim"
                    if st.button(
                        pin_text,
                        key=f"pin_{conv['id']}",
                        use_container_width=True,
                    ):
                        toggle_pin(conv["id"])
                        st.rerun()

                    if st.button(
                        "Đổi tên",
                        key=f"ren_{conv['id']}",
                        use_container_width=True,
                    ):
                        st.session_state.renaming_id = conv["id"]
                        st.rerun()

                    if st.button(
                        "Xóa",
                        key=f"del_{conv['id']}",
                        use_container_width=True,
                    ):
                        delete_conversation(conv["id"])
                        st.rerun()

    # ----- Hộp đổi tên -----
    if st.session_state.renaming_id:
        conv_id = st.session_state.renaming_id
        current = next(
            (c for c in st.session_state.conversations
             if c["id"] == conv_id),
            None,
        )
        if current:
            st.markdown(
                '<div class="side-heading">Đổi tên</div>',
                unsafe_allow_html=True,
            )
            new_name = st.text_input(
                "Tên mới",
                value=current["title"],
                key=f"rename_input_{conv_id}",
                label_visibility="collapsed",
            )
            c_save, c_cancel = st.columns([1, 1])
            with c_save:
                if st.button(
                    "Lưu", key="rename_save",
                    use_container_width=True,
                ):
                    rename_conversation(conv_id, new_name)
                    st.session_state.renaming_id = None
                    st.rerun()
            with c_cancel:
                if st.button(
                    "Hủy", key="rename_cancel",
                    use_container_width=True,
                ):
                    st.session_state.renaming_id = None
                    st.rerun()
        else:
            st.session_state.renaming_id = None


# ============================================================
# 10. CẢNH BÁO LỖI DB
# ============================================================

if not st.session_state.ready and "db_error" in st.session_state:
    st.error(
        "❌ **Không kết nối được Vector Database.**  \n"
        "Hãy chạy `python build_db.py` trước khi chạy app."
    )
    with st.expander("Chi tiết lỗi"):
        st.code(st.session_state.db_error)


# ============================================================
# 11. HERO
# ============================================================

def render_hero():
    st.markdown(
        f"""
        <div class="hero">
            {LOGO_HERO}
            <h1>
                Xin chào! Bạn đang quan tâm đến<br>
                <span class="accent">ngành và chuyên ngành nào?</span>
            </h1>
            <p>
                Trợ lý hỗ trợ tư vấn ngành, chuyên ngành,
                chương trình đào tạo và định hướng nghề nghiệp
                tại Khoa Công nghệ thông tin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


current_conv = get_current_conversation()

if (
    (current_conv is None or not current_conv["messages"])
    and not st.session_state.pending_question
):
    render_hero()


# ============================================================
# 12. HIỂN THỊ LỊCH SỬ CHAT
# ============================================================

if current_conv is not None:
    for message in current_conv["messages"]:

        role = message["role"]
        avatar = "👤" if role == "user" else ASSISTANT_AVATAR

        with st.chat_message(role, avatar=avatar):

            st.markdown(message["content"])

            if role == "assistant":
                render_sources(message.get("metadatas", []))

                if message.get("time"):
                    st.caption(
                        f"⏱️ Phản hồi trong {message['time']:.2f} giây"
                    )


# ============================================================
# 13. CHAT INPUT
# ============================================================

typed_prompt = st.chat_input("Nhắn tin cho Trợ lý Khoa CNTT...")


# ============================================================
# 14. XỬ LÝ CÂU HỎI
# ============================================================

question = None

if typed_prompt:
    question = typed_prompt
elif st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    process_question(question)
    st.rerun()