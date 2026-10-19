from dotenv import load_dotenv
load_dotenv()

import logging
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="QA Chatbot",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

SYSTEM_PROMPT = (
    "You are a clear, friendly assistant. Answer directly, then explain. "
    "Use short paragraphs, and use lists or code blocks only when they help. "
    "Use plain Markdown only. Never use HTML tags such as <br>."
)

SUGGESTIONS = [
    "Explain how DNS works, step by step",
    "Why does my Python code say 'list index out of range'?",
    "Give me 5 project ideas to learn machine learning",
]

# ---------------- CUSTOM CSS ----------------
# Palette: ink green #0f1514 / panel #172120 / line #28352f
#          text #e7eeea / muted #8aa097 / butter #f0d98a
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&display=swap');

:root {
    --bg: #0f1514;
    --panel: #172120;
    --line: #28352f;
    --text: #e7eeea;
    --muted: #8aa097;
    --accent: #f0d98a;
    --accent-ink: #1d1a0c;
}

/* Base */
.stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

/* Reading column */
.block-container {
    max-width: 760px !important;
    padding: 3rem 1.25rem 8rem !important;
}

/* Welcome */
.welcome h1 {
    font-family: 'Source Serif 4', Georgia, serif;
    font-weight: 500;
    font-size: clamp(34px, 6vw, 52px);
    line-height: 1.12;
    letter-spacing: -0.02em;
    color: var(--text);
    margin: 8vh 0 14px;
    padding: 0;
}
.welcome p {
    color: var(--muted);
    font-size: 17px;
    line-height: 1.6;
    margin: 0 0 28px;
    max-width: 52ch;
}

/* Suggestion buttons */
.stApp div.stButton > button {
    width: 100%;
    min-height: 92px;
    padding: 14px 16px;
    text-align: left;
    justify-content: flex-start;
    align-items: flex-start;
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    transition: border-color .15s ease, background .15s ease;
}
.stApp div.stButton > button p {
    color: var(--text) !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    line-height: 1.45 !important;
    text-align: left !important;
}
.stApp div.stButton > button:hover { border-color: var(--accent) !important; background: #1c2927 !important; }
.stApp div.stButton > button:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }

/* Messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
    gap: 14px !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    line-height: 1.75;
    font-size: 16px;
}

/* Assistant: open prose in a serif, no box */
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) [data-testid="stChatMessageContent"] {
    font-family: 'Source Serif 4', Georgia, serif;
    color: var(--text);
    font-size: 17px;
}
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) p,
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) li {
    font-size: 17px;
}

/* User: right-aligned bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: var(--accent);
    color: var(--accent-ink);
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    width: fit-content;
    max-width: 82%;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
    color: var(--accent-ink) !important;
    margin: 0;
}

/* Avatars */
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"] {
    background: var(--panel) !important;
    border: 1px solid var(--line);
    color: var(--accent) !important;
}

/* Code blocks */
[data-testid="stChatMessage"] pre, [data-testid="stCode"] {
    background: #0a0f0e !important;
    border: 1px solid var(--line);
    border-radius: 12px;
}

/* Bottom bar */
[data-testid="stBottom"], [data-testid="stBottom"] > div {
    background: linear-gradient(to top, var(--bg) 70%, transparent) !important;
}
[data-testid="stBottomBlockContainer"] {
    max-width: 760px !important;
    padding-bottom: 1.5rem !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--accent) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"] div { background-color: transparent !important; }
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    color: var(--accent-ink) !important;
    border-radius: 10px !important;
    border: none !important;
}
[data-testid="stChatInput"] button:hover { filter: brightness(1.08); }
[data-testid="stChatInput"] button:disabled { background: var(--line) !important; color: var(--muted) !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebar"] .stButton > button {
    min-height: 42px; align-items: center; justify-content: center;
    background: transparent !important; border: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] .stButton > button p { text-align: center !important; }
.side-note { color: var(--muted) !important; font-size: 13px; line-height: 1.5; }

/* Spinner */
[data-testid="stSpinner"] p { color: var(--muted) !important; }

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
}

/* Mobile */
@media (max-width: 640px) {
    .block-container { padding: 2rem 1rem 8rem !important; }
    .welcome h1 { margin-top: 3vh; }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] { max-width: 92%; }
}
</style>
""", unsafe_allow_html=True)


# ---------------- TEXT CLEANUP ----------------
import re

BR_TAG = re.compile(r"<br\s*/?>", re.IGNORECASE)

def clean_text(text: str) -> str:
    """Turn stray <br> tags into something Markdown can display."""
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            # Inside a table row a newline would break the table
            line = BR_TAG.sub(" ", line)
        else:
            line = BR_TAG.sub("\n", line)
        out.append(line)
    text = "\n".join(out)
    # hide a half-streamed tag like "<b" at the very end
    return re.sub(r"<[^>\n]{0,5}$", "", text)


# ---------------- LLM ----------------
@st.cache_resource
def get_llm(temperature: float):
    return ChatGroq(model="openai/gpt-oss-120b", temperature=temperature)


# ---------------- SESSION STATE ----------------
def new_chat():
    st.session_state.history = [SystemMessage(content=SYSTEM_PROMPT)]
    st.session_state.messages = []

if "history" not in st.session_state or "messages" not in st.session_state:
    new_chat()


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ✦ QA Chatbot")
    if st.button("New chat", use_container_width=True):
        new_chat()
        st.rerun()

    temperature = st.slider(
        "Creativity",
        min_value=0.0, max_value=1.0, value=0.7, step=0.1,
        help="Lower gives more focused answers. Higher gives more varied ones.",
    )
    st.markdown(
        f"<p class='side-note'>{len(st.session_state.messages)} messages in this chat</p>",
        unsafe_allow_html=True,
    )

llm = get_llm(temperature)


# ---------------- WELCOME ----------------
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <h1>What do you want to know?</h1>
        <p>Ask a question, paste some code, or start from one of these.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(len(SUGGESTIONS))
    for col, text in zip(cols, SUGGESTIONS):
        with col:
            if st.button(text, key=f"sugg_{text}", use_container_width=True):
                st.session_state.pending_query = text
                st.rerun()


# ---------------- CHAT HISTORY ----------------
AVATARS = {"user": "🙂", "assistant": "✨"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(clean_text(message["content"]))


# ---------------- CHAT INPUT ----------------
query = st.chat_input("Ask a question…")

if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")


# ---------------- GENERATE RESPONSE ----------------
def stream_answer(history):
    for chunk in llm.stream(history):
        if chunk.content:
            yield chunk.content


if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.history.append(HumanMessage(content=query))

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(query)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            placeholder = st.empty()
            answer = ""
            with st.spinner("Thinking…"):
                stream = stream_answer(st.session_state.history)
                first = next(stream, "")
            answer = first
            placeholder.markdown(clean_text(answer) + "▌")
            for piece in stream:
                answer += piece
                placeholder.markdown(clean_text(answer) + "▌")
            answer = clean_text(answer)
            placeholder.markdown(answer)

            st.session_state.history.append(AIMessage(content=answer))
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception:
            st.session_state.history.pop()
            st.session_state.messages.pop()
            st.error("Couldn't get a response. Check your connection and API key, then send your message again.")
            logging.exception("QA chatbot response generation failed")