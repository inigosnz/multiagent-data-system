import streamlit as st
from agents.query_extractor import extract_conditions
from agents.json_formatter import format_to_json
from agents.pandas_coder import generate_pandas_code
from agents.code_verifier import verify_code
from agents.code_executor import execute_pandas_code

# ─────────────────────────────────────────────────────────
MAX_RETRIES = 10

# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Agent Pandas Assistant")
st.title("🤖 Multi-Agent Pandas Query System")

query = st.text_input("🔍 Enter your query:", value="")
verbose = st.checkbox("🔊 Verbose mode", value=True)

if st.button("Run"):
    with st.spinner("Running agents..."):

        # Extract conditions
        extracted = extract_conditions(query)
        if verbose:
            st.subheader("🧠 Extracted Conditions")
            st.code(extracted)

        #  Convert to JSON
        json_query = format_to_json(extracted)
        if verbose:
            st.subheader("📜 JSON Query")
            st.code(json_query, language="json")

        #  Generate + Verify Code with retries
        error_msg = ""
        for attempt in range(1, MAX_RETRIES + 1):
            if verbose:
                st.markdown(f"🔁 **Attempt {attempt}**")

            code = generate_pandas_code(json_query, error_msg)
            valid, err = verify_code(code)

            if valid:
                break
            else:
                error_msg = f"The last code failed with error:\n{err}"
                if attempt == MAX_RETRIES:
                    st.error("❌ All code generation attempts failed.")
                    st.info(f"🧠 LLM Error Insight:\n{err}")
                    st.stop()

        # Show and Execute Code
        st.subheader("🐍 Final Pandas Code")
        st.code(code, language="python")
        st.subheader("📊 Output Table")
        execute_pandas_code(code)

