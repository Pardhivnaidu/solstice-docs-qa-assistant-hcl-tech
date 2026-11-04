import streamlit as st
import pandas as pd
import rag_backend as rag


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Solstice Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "messages": [],
    "evaluation_report": None,
    "diagnostic_result": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# BACKEND INITIALIZATION
# ============================================================

@st.cache_resource(show_spinner="Loading documents and search indexes...")
def initialize_backend():
    return rag.initialize_knowledge_base()


try:
    kb_info = initialize_backend()
    backend_error = None
except Exception as exc:
    kb_info = None
    backend_error = str(exc)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_text(value, default="Not available"):
    """Return a display-safe string."""
    if value is None or value == "":
        return default
    return str(value)


def get_status_label(status):
    labels = {
        "answered": "Answered",
        "not_found": "Not found in documentation",
        "needs_clarification": "Needs clarification",
        "conflict": "Conflicting documentation",
        "source_excerpt": "Original documentation excerpts",
    }
    return labels.get(status, safe_text(status, "Unknown status"))


def render_result(result):
    """Render one answer result using native Streamlit components."""
    if not isinstance(result, dict):
        st.error("The backend returned an unexpected response.")
        st.write(result)
        return

    status = result.get("status", "unknown")
    answer = result.get("answer")

    if status == "answered":
        st.success(get_status_label(status))
    elif status in ("not_found", "conflict"):
        st.warning(get_status_label(status))
    elif status == "needs_clarification":
        st.info(get_status_label(status))
    else:
        st.info(get_status_label(status))

    if answer:
        st.markdown(str(answer))
    else:
        st.write("No answer text was returned.")

    clarification = result.get("clarification_question")
    if clarification:
        st.info(f"Clarification: {clarification}")

    conflict_details = result.get("conflict_details")
    if conflict_details:
        st.warning(f"Conflict details: {conflict_details}")

    sources = result.get("sources") or []
    if sources:
        st.markdown("**Sources**")
        st.write(", ".join(str(source) for source in sources))

    quotes = result.get("quotes") or []
    if quotes:
        with st.expander("Supporting quotations", expanded=True):
            for quote in quotes:
                st.markdown(f"> {quote}")

    excerpts = result.get("excerpts") or []
    if excerpts:
        with st.expander("Original documentation excerpts", expanded=True):
            for excerpt in excerpts:
                if isinstance(excerpt, dict):
                    doc_id = excerpt.get("doc_id", "Unknown document")
                    title = excerpt.get("title", "")
                    text = excerpt.get("text", "")

                    st.markdown(f"**{doc_id} — {title}**")
                    st.write(text)
                else:
                    st.write(excerpt)

    verification = result.get("verification") or {}

    if isinstance(verification, dict) and verification:
        if verification.get("passed"):
            st.caption("Answer verification: Passed")
        else:
            st.caption(
                "Answer verification: "
                + safe_text(verification.get("message"), "Not passed")
            )

    generation_error = result.get("generation_error")

    if generation_error:
        with st.expander("Generation error details"):
            st.code(str(generation_error))


def render_backend_error():
    st.error("The knowledge base could not be initialized.")

    st.code(backend_error or "Unknown initialization error")

    st.caption(
        "Check that rag_backend.py, the documents JSON file, dependencies, "
        "and your .env configuration are present and valid."
    )


def display_table(rows):
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No results are available.")


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📚 Solstice Assistant")

page = st.sidebar.radio(
    "Navigation",
    [
        "Ask a Question",
        "Evaluation Dashboard",
        "Retrieval Diagnostics",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.subheader("System status")

if backend_error:
    st.sidebar.error("Knowledge base unavailable")
else:
    st.sidebar.success("Knowledge base ready")

    if isinstance(kb_info, dict):
        st.sidebar.metric(
            "Documents",
            kb_info.get("document_count", "—"),
        )

        st.sidebar.metric(
            "Chunks",
            kb_info.get("chunk_count", "—"),
        )
    else:
        st.sidebar.caption("Knowledge base initialized.")

st.sidebar.caption(
    f"Embedding model: {getattr(rag, 'EMBEDDING_MODEL', 'Configured in backend')}"
)

st.sidebar.caption(
    f"Gemini model: {getattr(rag, 'GEMINI_MODEL', 'Configured in backend')}"
)


# Clear chat history

if st.sidebar.button("Clear chat history", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


# Rebuild knowledge base

if st.sidebar.button("Rebuild knowledge base", use_container_width=True):
    try:
        with st.spinner("Rebuilding the knowledge base..."):
            rag.initialize_knowledge_base(force_rebuild=True)

        initialize_backend.clear()

        st.sidebar.success("Knowledge base rebuilt.")
        st.rerun()

    except Exception as exc:
        st.sidebar.error(f"Rebuild failed: {exc}")


# ============================================================
# PAGE 1: ASK A QUESTION
# ============================================================

if page == "Ask a Question":

    st.title("📚 Solstice Learning Documentation Assistant")

    st.subheader("Find answers in the Solstice Learning documentation")

    st.write(
        "Ask a question about the platform. Answers are grounded in the "
        "available documentation, and the assistant abstains when the "
        "documentation does not provide sufficient evidence."
    )

    st.divider()

    if backend_error:
        render_backend_error()
        st.stop()

    # Display conversation history

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            if message["role"] == "user":
                st.markdown(message["content"])

            else:
                render_result(message["result"])

    # User input

    question = st.chat_input(
        "Ask something about Solstice Learning..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer

        with st.chat_message("assistant"):

            try:
                with st.spinner(
                    "Searching documentation and preparing an answer..."
                ):
                    result = rag.answer_question(question)

                render_result(result)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "result": result,
                    }
                )

            except Exception as exc:
                st.error(f"Something went wrong: {exc}")


# ============================================================
# PAGE 2: EVALUATION DASHBOARD
# ============================================================

elif page == "Evaluation Dashboard":

    st.title("🧪 Evaluation Dashboard")

    st.write(
        "Run the five existing evaluation cases from "
        "`example_questions.json` against the RAG pipeline."
    )

    st.divider()

    if backend_error:
        render_backend_error()
        st.stop()

    # Run evaluation

    if st.button("Run Evaluation", type="primary"):

        try:
            with st.spinner("Running evaluation cases..."):
                st.session_state.evaluation_report = rag.run_evaluation()

        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    report = st.session_state.evaluation_report

    if report:

        total = report.get("total", 0)
        passed = report.get("passed", 0)
        failed = report.get("failed", max(total - passed, 0))
        accuracy = report.get("accuracy", 0)

        # Metrics

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total cases", total)
        col2.metric("Passed", passed)
        col3.metric("Failed", failed)
        col4.metric("Accuracy", f"{accuracy}%")

        st.progress(
            min(max(float(accuracy) / 100, 0.0), 1.0)
            if total
            else 0.0
        )

        # Results table

        st.subheader("Test case results")

        rows = []

        for item in report.get("results", []):

            rows.append(
                {
                    "Case": item.get("question_id"),
                    "Question": item.get("question"),
                    "Expected document": item.get("expected_doc_id") or "None",
                    "Expected status": item.get("expected_status"),
                    "Actual status": item.get("actual_status"),
                    "Retrieval": (
                        "PASS" if item.get("retrieval_correct") else "FAIL"
                    ),
                    "Source": (
                        "PASS" if item.get("source_correct") else "FAIL"
                    ),
                    "Status": (
                        "PASS" if item.get("status_correct") else "FAIL"
                    ),
                    "Answer": (
                        "PASS" if item.get("answer_correct") else "FAIL"
                    ),
                    "Overall": (
                        "PASS" if item.get("passed") else "FAIL"
                    ),
                }
            )

        display_table(rows)

        # Individual case details

        st.subheader("Detailed results")

        for item in report.get("results", []):

            passed_case = bool(item.get("passed"))
            label = "PASS" if passed_case else "FAIL"

            with st.expander(
                f"{item.get('question_id', 'Case')} — {label}"
            ):

                st.markdown(
                    f"**Question:** {safe_text(item.get('question'))}"
                )

                st.markdown(
                    f"**Answer:** {safe_text(item.get('answer'))}"
                )

                st.markdown(
                    f"**Expected status:** "
                    f"`{safe_text(item.get('expected_status'))}`"
                )

                st.markdown(
                    f"**Actual status:** "
                    f"`{safe_text(item.get('actual_status'))}`"
                )

                diagnostics = item.get("evidence_diagnostics")

                if diagnostics is not None:
                    st.markdown("**Evidence Gate diagnostics**")
                    st.json(diagnostics)


# ============================================================
# PAGE 3: RETRIEVAL DIAGNOSTICS
# ============================================================

elif page == "Retrieval Diagnostics":

    st.title("🔍 Retrieval Diagnostics")

    st.write(
        "Inspect the retrieval pipeline: vector search, BM25 keyword search, "
        "Reciprocal Rank Fusion (RRF), and the Evidence Gate."
    )

    st.divider()

    if backend_error:
        render_backend_error()
        st.stop()

    diagnostic_question = st.text_input(
        "Enter a question",
        placeholder="How do I reset my password?",
    )

    # Inspect retrieval

    if st.button(
        "Inspect Retrieval",
        type="primary",
        disabled=not diagnostic_question.strip(),
    ):

        try:
            with st.spinner("Inspecting retrieval results..."):

                st.session_state.diagnostic_result = rag.answer_question(
                    diagnostic_question.strip()
                )

        except Exception as exc:
            st.error(f"Diagnostics failed: {exc}")

    result = st.session_state.diagnostic_result

    if result:

        st.divider()

        # Evidence Gate metrics

        gate = result.get("evidence_gate") or {}

        if not isinstance(gate, dict):
            gate = {}

        col1, col2, col3 = st.columns(3)

        similarity = gate.get("similarity")
        coverage = gate.get("keyword_coverage")

        col1.metric(
            "Best similarity",
            (
                f"{float(similarity):.4f}"
                if isinstance(similarity, (int, float))
                else "—"
            ),
        )

        col2.metric(
            "Keyword coverage",
            (
                f"{float(coverage):.0%}"
                if isinstance(coverage, (int, float))
                else "—"
            ),
        )

        col3.metric(
            "Evidence Gate",
            "PASS" if gate.get("passed") else "BLOCKED",
        )

        if gate.get("reason"):
            st.caption(gate["reason"])

        # Per-chunk decisions

        st.subheader("Evidence Gate decisions")

        per_chunk = gate.get("per_chunk") or []

        if per_chunk:

            st.dataframe(
                pd.DataFrame(per_chunk),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info("No per-chunk evidence decisions were returned.")

        # Vector search

        st.subheader("Vector search results")

        with st.expander("View vector search", expanded=False):
            display_table(result.get("vector_results") or [])

        # BM25 search

        st.subheader("BM25 search results")

        with st.expander("View BM25 results", expanded=False):
            display_table(result.get("bm25_results") or [])

        # RRF results

        st.subheader("Final RRF rankings")

        with st.expander("View fused retrieval results", expanded=False):
            display_table(result.get("retrieved_documents") or [])

        # Final answer

        st.divider()

        st.subheader("Assistant response")

        render_result(result)