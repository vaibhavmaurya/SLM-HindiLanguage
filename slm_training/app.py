"""Streamlit testing UI for Hindi SLM.

Flow:
  1. Enter English text
  2. Translate to Hindi via Ollama (configurable model)
  3. Feed Hindi text to vaibhavmaurya/hindi-slm-v001 for continuation
  4. Always shows input/output token counts

Run:
  cd slm_training
  streamlit run app.py
"""

import json
import sys
import time
from pathlib import Path

import requests
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ────────────────────────────────────────────────────────────────

HF_REPO_ID   = "vaibhavmaurya/hindi-slm-v001"
OLLAMA_URL   = "http://localhost:11434"

TRANSLATE_SYSTEM = (
    "You are a precise English-to-Hindi translator. "
    "Translate the user's sentence into natural, fluent Hindi using Devanagari script. "
    "Output ONLY the Hindi translation — no explanations, no Roman script."
)


# ── Cached resources ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading tokenizer from Hub...")
def load_tokenizer():
    return AutoTokenizer.from_pretrained(HF_REPO_ID)


@st.cache_resource(show_spinner="Loading Hindi SLM from Hub...")
def load_model():
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = AutoModelForCausalLM.from_pretrained(
        HF_REPO_ID,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, torch.device(device)


# ── Ollama helpers ────────────────────────────────────────────────────────────

def list_ollama_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def translate_stream(text: str, model_name: str):
    """Yield translation chunks via Ollama chat streaming."""
    payload = {
        "model":    model_name,
        "messages": [
            {"role": "system", "content": TRANSLATE_SYSTEM},
            {"role": "user",   "content": text},
        ],
        "stream": True,
    }
    with requests.post(
        f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=60
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            yield chunk.get("message", {}).get("content", "")
            if chunk.get("done"):
                break


# ── SLM generation ────────────────────────────────────────────────────────────

def generate_hindi(
    prompt: str,
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple[str, int, int, float]:
    """Returns (generated_text, n_input_tokens, n_output_tokens, elapsed_sec)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    n_input = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    new_ids   = output_ids[0, n_input:]
    n_output  = new_ids.shape[0]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    return generated, n_input, n_output, elapsed


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Hindi SLM Tester", page_icon="🇮🇳", layout="wide")
st.title("🇮🇳 Hindi SLM — Testing UI")
st.caption(f"Model: `{HF_REPO_ID}` · 46M params · bfloat16")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    # Ollama model picker
    ollama_models = list_ollama_models()
    if ollama_models:
        ollama_model = st.selectbox("Translation model (Ollama)", ollama_models)
        st.success("Ollama connected")
    else:
        st.error("Ollama not reachable at localhost:11434")
        ollama_model = st.text_input("Model name (manual)", value="llama3.2:latest")

    st.divider()
    st.subheader("Generation params")
    max_new_tokens     = st.slider("Max new tokens",     20,  512, 150, step=10)
    temperature        = st.slider("Temperature",       0.1,  1.5, 0.8, step=0.05)
    top_p              = st.slider("Top-p",             0.5,  1.0, 0.9, step=0.05)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.1, step=0.05)

    st.divider()

    # Model status
    device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"Inference device: **{device_label}**")
    st.caption(f"Hub repo: `{HF_REPO_ID}`")


# ── Main layout ───────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2)

# ── Left: English input + translation ────────────────────────────────────────

with col_left:
    st.subheader("English Input")
    english_text = st.text_area(
        "English text",
        height=140,
        placeholder="Type or paste English text here...",
        label_visibility="collapsed",
    )

    translate_btn = st.button(
        "🔄 Translate to Hindi",
        disabled=not english_text.strip(),
        use_container_width=True,
        type="primary",
    )

    st.subheader("Hindi Text (editable)")
    if "hindi_text" not in st.session_state:
        st.session_state.hindi_text = ""

    if translate_btn and english_text.strip():
        stream_placeholder = st.empty()
        collected = ""
        try:
            for chunk in translate_stream(english_text.strip(), ollama_model):
                collected += chunk
                stream_placeholder.text_area(
                    "live_hindi",
                    value=collected,
                    height=140,
                    label_visibility="collapsed",
                )
            st.session_state.hindi_text = collected.strip()
            stream_placeholder.empty()
        except Exception as e:
            st.error(f"Ollama error: {e}")

    hindi_text = st.text_area(
        "Hindi text",
        value=st.session_state.hindi_text,
        height=140,
        key="hindi_input_area",
        label_visibility="collapsed",
        placeholder="Hindi translation will appear here (editable)...",
    )

    # Live input token count
    if hindi_text.strip():
        tokenizer = load_tokenizer()
        n_input_preview = len(tokenizer.encode(hindi_text, add_special_tokens=True))
        st.metric("Input tokens (to SLM)", n_input_preview, help="Tokens that will be fed to the SLM")
    else:
        st.metric("Input tokens (to SLM)", 0)


# ── Right: SLM generation output ─────────────────────────────────────────────

with col_right:
    st.subheader("SLM Generated Hindi")

    generate_btn = st.button(
        "⚡ Generate with SLM",
        disabled=not hindi_text.strip(),
        use_container_width=True,
        type="primary",
    )

    output_placeholder = st.empty()

    if "generated_text" not in st.session_state:
        st.session_state.generated_text = ""
        st.session_state.n_in  = 0
        st.session_state.n_out = 0
        st.session_state.elapsed = 0.0

    if generate_btn and hindi_text.strip():
        model, device = load_model()
        tokenizer     = load_tokenizer()

        with st.spinner("Generating..."):
            gen, n_in, n_out, elapsed = generate_hindi(
                hindi_text.strip(),
                model,
                tokenizer,
                device,
                max_new_tokens,
                temperature,
                top_p,
                repetition_penalty,
            )

        st.session_state.generated_text = gen
        st.session_state.n_in    = n_in
        st.session_state.n_out   = n_out
        st.session_state.elapsed = elapsed

    output_placeholder.text_area(
        "Generated output",
        value=st.session_state.generated_text,
        height=280,
        label_visibility="collapsed",
        placeholder="SLM continuation will appear here...",
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Input tokens",  st.session_state.n_in)
    m2.metric("Output tokens", st.session_state.n_out)
    if st.session_state.elapsed > 0:
        tok_per_sec = st.session_state.n_out / st.session_state.elapsed
        m3.metric("Speed", f"{tok_per_sec:.1f} tok/s")
    else:
        m3.metric("Speed", "—")


# ── Full output expander ──────────────────────────────────────────────────────

if st.session_state.generated_text and hindi_text.strip():
    with st.expander("📄 Full output (prompt + continuation)"):
        st.text(hindi_text.strip() + " " + st.session_state.generated_text)


# ── Token debug expander ──────────────────────────────────────────────────────

if hindi_text.strip():
    with st.expander("🔍 Token debug — see how Hindi text is tokenized"):
        tokenizer = load_tokenizer()
        ids    = tokenizer.encode(hindi_text.strip(), add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        rows   = [f"`{tok}` → {tid}" for tok, tid in zip(tokens, ids)]
        c1, c2, c3 = st.columns(3)
        for i, row in enumerate(rows):
            [c1, c2, c3][i % 3].markdown(row)
