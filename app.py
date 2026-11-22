"""Streamlit app: upload files or paste text, get summaries.   Run: streamlit run app.py"""

import hashlib
import html
from pathlib import Path

import streamlit as st

from summarizer.abstractive import load_model
from summarizer.export import summary_points, to_excel, to_txt
from summarizer.extractive import load_embedding_model
from summarizer.pipeline import read_file, summarize_document
from summarizer.preprocessing import split_words

ASSETS = Path(__file__).parent / "assets"
SAMPLE_TEXT = (ASSETS / "sample.txt").read_text(encoding="utf-8")

# Fingerprint of the summarizer code: when the code changes, old cached results are not reused
CODE_VERSION = hashlib.md5(
    b"".join(p.read_bytes() for p in sorted((Path(__file__).parent / "summarizer").glob("*.py")))
).hexdigest()

st.set_page_config(page_title="Text Summarizer", page_icon="✨", layout="wide")

st.markdown(f"<style>{(ASSETS / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# st.cache_resource keeps the models in memory between button clicks
@st.cache_resource(show_spinner="Loading the sentence-embedding model...")
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource(show_spinner="Loading DistilBART (first run downloads ~1.2 GB)...")
def get_bart():
    return load_model()


# st.cache_data remembers results: same input + same settings -> instant answer
@st.cache_data(show_spinner=False, max_entries=500)
def cached_read(name, data, code_version):
    return read_file(name, data)


@st.cache_data(show_spinner=False, max_entries=500)
def cached_summary(raw_text, method, length, ratio, code_version):
    return summarize_document(raw_text, method, length, ratio)


def load_sample():
    st.session_state["pasted"] = SAMPLE_TEXT


def clear_all():
    st.session_state["pasted"] = ""
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.session_state.pop("results", None)


EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# --- Header ---
st.markdown(
    """
<div class="hero">
  <div>
    <h1>✨ Turn lengthy text into clear insights.</h1>
    <p>Summarize articles, documents and more with the power of AI.</p>
  </div>
  <div class="robot">🤖</div>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.15], gap="medium")


# --- Left card: input and settings ---
with left:
    with st.container(key="content_card"):
        st.markdown('<p class="card-title">📄 Your Content</p>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Drag & drop files here: any number of PDF, TXT or DOCX files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.get('uploader_key', 0)}",
        )
        if uploaded_files:
            st.caption(f"📚 {len(uploaded_files)} file(s) selected. Each one gets its own summary.")

        st.markdown(
            "<p style='text-align:center;color:#9ca3af;font-size:0.85rem;margin:0.3rem 0'>"
            "— or paste your text below —</p>",
            unsafe_allow_html=True,
        )
        pasted = st.text_area(
            "Your text",
            key="pasted",
            height=170,
            placeholder="Paste your article, report or any text here...",
            label_visibility="collapsed",
        )

        words_col, sample_col = st.columns([1, 1])
        words_col.caption(f"Aa  {len(split_words(pasted)):,} words")
        sample_col.button("✨ Try a sample", on_click=load_sample, width="stretch")

        st.markdown('<p class="section-title">⚙️ Summary Preferences</p>', unsafe_allow_html=True)

        method = st.segmented_control(
            "🧠 Method", ["Extractive", "Abstractive"], default="Extractive", width="stretch"
        ) or "Extractive"
        st.markdown(
            '<div class="hint"><span>Picks key sentences · fast</span>'
            "<span>Writes new sentences · AI model</span></div>",
            unsafe_allow_html=True,
        )

        length = st.segmented_control(
            "📏 Summary Length", ["Short", "Medium", "Long"], default="Medium", width="stretch"
        ) or "Medium"
        st.markdown(
            '<div class="hint"><span>Key points only</span><span>Balanced summary</span>'
            "<span>Detailed summary</span></div>",
            unsafe_allow_html=True,
        )

        use_custom = st.toggle("🎚️ Custom length")
        custom_pct = st.slider(
            "Summary size (% of the original text)",
            min_value=10, max_value=90, value=30, step=5, format="%d%%",
            disabled=not use_custom,
        )

        style = st.segmented_control(
            "📝 Summary Style", ["Paragraph", "Bullet Points"], default="Paragraph", width="stretch"
        ) or "Paragraph"

        generate = st.button("✨ Generate Summary  ›", key="generate", width="stretch")

# --- Summarize every document ---
if generate:
    # Every uploaded file is one document; pasted text is one more
    documents = [(f.name, f.getvalue()) for f in uploaded_files or []]
    if pasted.strip():
        documents.append(("Pasted text", pasted))

    if not documents:
        with left:
            st.warning("Please upload files, paste some text, or click **Try a sample**.")
    else:
        settings = (method.lower(), length.lower(), custom_pct / 100 if use_custom else None)

        with right:
            # Load the models once, before the loop
            get_embedding_model()
            if method == "Abstractive":
                get_bart()

            progress = st.progress(0.0)
            results = []
            for i, (name, content) in enumerate(documents):
                progress.progress(i / len(documents), text=f"Summarizing {i + 1} of {len(documents)}: {name}")
                try:
                    raw = content if isinstance(content, str) else cached_read(name, content, CODE_VERSION)
                    result = cached_summary(raw, *settings, CODE_VERSION)
                    results.append({"name": name, "method": method, **result})
                except Exception as error:  # one bad file must not stop the others
                    results.append({"name": name, "error": str(error) or type(error).__name__})
            progress.empty()

        st.session_state["results"] = results
        st.session_state["selected_doc"] = results[0]["name"]

# --- Right card: results ---
with right:
    with st.container(key="summary_card"):
        title_col, copy_col, dl_col, clear_col = st.columns([2.2, 1, 1.2, 0.6], vertical_alignment="center")
        title_col.markdown('<p class="card-title">✨ AI Summary</p>', unsafe_allow_html=True)

        results = st.session_state.get("results")

        if not results:
            st.markdown(
                '<div class="empty"><div class="big">🤖</div>'
                "<p>Your summary will appear here.<br>Upload one or more files or paste text, "
                "then click <b>Generate Summary</b>.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            clear_col.button("↻", on_click=clear_all, help="Clear and start again", width="stretch")

            # Overview table (only when there are several documents)
            if len(results) > 1:
                done = [r for r in results if "summary" in r]
                st.markdown(
                    f'<div class="panel"><h4>📚 {len(done)} of {len(results)} documents summarized</h4></div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    [
                        {
                            "File": r["name"],
                            "Original words": r.get("original_words"),
                            "Summary words": r.get("summary_words"),
                            "Reduced": (f"{100 - round(100 * r['summary_words'] / max(r['original_words'], 1))}%"
                                        if "summary" in r else "—"),
                            "Status": "✅ Done" if "summary" in r else f"⚠️ {r['error']}",
                        }
                        for r in results
                    ],
                    hide_index=True,
                    width="stretch",
                )
                all_xlsx, all_txt = st.columns(2)
                all_xlsx.download_button("⬇️ All summaries (Excel)", to_excel(results),
                                        file_name="summaries.xlsx", mime=EXCEL_MIME, width="stretch")
                all_txt.download_button("⬇️ All summaries (TXT)", to_txt(results, style),
                                        file_name="summaries.txt", width="stretch")

                names = [r["name"] for r in results]
                chosen_name = st.selectbox("🔎 Show details for", names, key="selected_doc")
                result = results[names.index(chosen_name)]
            else:
                result = results[0]

            # Details of the selected document
            if "error" in result:
                st.error(f"**{result['name']}** could not be summarized: {result['error']}")
            else:
                summary = result["summary"]

                with copy_col.popover("📋 Copy", width="stretch"):
                    st.caption("Click the copy icon in the top-right corner of the box.")
                    st.code(summary, language=None, wrap_lines=True)
                dl_col.download_button("⬇️ Download", to_txt([result], style), width="stretch",
                                       file_name=f"{result['name'].rsplit('.', 1)[0]}_summary.txt")

                if style == "Bullet Points":
                    items = "".join(f"<li>{html.escape(s)}</li>" for s in summary_points(summary))
                    body = f"<ul>{items}</ul>"
                else:
                    body = f"<p>{html.escape(summary)}</p>"

                points = "".join(f"<li>{html.escape(p)}</li>" for p in result["key_points"])
                chips = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in result["topics"])

                original, short = result["original_words"], result["summary_words"]
                reduction = max(0, 100 - round(100 * short / max(original, 1)))
                file_label = html.escape(result["name"]) if len(results) > 1 else result["method"]

                st.markdown(
                    f"""
<div class="panel"><h4>📄 Summary <small style="color:#8b7bff;font-weight:500">· {file_label}</small></h4>{body}</div>
<div class="grid-2">
  <div class="panel"><h4>💡 Key Points</h4><ul class="check">{points}</ul></div>
  <div class="panel"><h4>🏷️ Important Topics</h4>{chips}</div>
</div>
<div class="panel">
  <h4>📊 Summary Reduction <span class="pill">{reduction}% Reduced 🚀</span></h4>
  <div class="bar"><div style="width:{reduction}%"></div></div>
  <div class="stats">
    <div><small>Original Words</small><b>{original:,}</b></div>
    <div><small>Summarized Words</small><b>{short:,}</b></div>
    <div><small>Words Reduced</small><b>{original - short:,}</b></div>
  </div>
</div>
<div class="footer-note">⭐ A shorter read. A clearer mind. ✨</div>
""",
                    unsafe_allow_html=True,
                )

                # For extractive: highlight the chosen sentences inside the original text
                if result["selected"]:
                    with st.expander("🔍 See which sentences were chosen"):
                        chosen = set(result["selected"])
                        marked = " ".join(
                            f"<mark>{html.escape(s)}</mark>" if s in chosen else html.escape(s)
                            for s in result["sentences"]
                        )
                        st.markdown(f'<div class="original-text">{marked}</div>', unsafe_allow_html=True)
