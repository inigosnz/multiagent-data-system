import streamlit as st
import pandas as pd

from agents.query_extractor import extract_conditions
from agents.json_formatter import format_to_json
from agents.pandas_coder import generate_pandas_code
from agents.code_verifier import verify_code
from agents.code_executor import execute_pandas_code
from agents.result_summarizer import summarize_result

# ─────────────────────────────────────────────────────────
MAX_RETRIES = 10

# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Agent Pandas Assistant")
st.title("🤖 Multi-Agent Pandas Query System")

#Guide the user
with st.expander("ℹ️ How to Use This App"):
    st.markdown("""
    **Step-by-step guide:**

    1. **Upload your CSV file** in the sidebar under "CSV File".
    2. **Upload a description file** in `.txt` format that explains your dataset.

        - The file **must be plain text (`.txt`)**.
        - It should clearly describe the dataset structure, variables, and any relevant logic or rules.
        - Follow a format like this:

            ```
            # Dataset Description: your_dataset.csv

            ## Main Variables
            - **Column A**: Short description of what this column represents.
            - **Column B**: Details about expected values or formats.
            - **Column C**: Units of measurement, if applicable.
            - ...

            ## Domain-Specific Rules or Logic
            1. Condition A — Description of condition
            Trigger Condition: Explain the criteria or thresholds.
            ...
            2. Condition B — Description of another logic or case
            Trigger Condition: Define what causes this condition.
            ...
            ```

        - Make sure to include any **business logic**, **domain rules**, or **target label definitions** in this section.

    3. **Enter your query** in natural language:
        - _"Find rows where Condition A is likely to occur."_
        - _"Show average values grouped by Column B."_

    4. Toggle **Verbose Mode** to see all internal steps taken by the assistant.

    5. Click **Run** to process the request with AI-driven agents.
    """)

summary_mode = st.radio(
    "🧠 Choose summary style:",
    ["Short", "Detailed", "Stats-heavy"],
    horizontal=True
)


# Sidebar inputs
st.sidebar.header("🧾 Upload Your Files")
uploaded_csv = st.sidebar.file_uploader("📁 CSV File", type=["csv"])
uploaded_description = st.sidebar.file_uploader("📝 Description File", type=["txt"])

st.sidebar.header("🔧 Settings")
verbose = st.sidebar.checkbox("🔊 Verbose mode", value=True)

# Main area
query = st.text_input("🔍 Enter your query:", value="")


# ─────────────────────────────────────────────────────────
if st.button("Run") and uploaded_csv and uploaded_description:
    with st.spinner("Loading and processing..."):

        # Load CSV + Description
        df = pd.read_csv(uploaded_csv)
        dataset_description = uploaded_description.read().decode("utf-8")

        # STEP 1 — Extract conditions
        extracted = extract_conditions(query, dataset_description)
        if verbose:
            st.subheader("🧠 Extracted Conditions")
            st.code(extracted)

        # STEP 2 — Format to JSON
        json_query = format_to_json(extracted, dataset_description)
        if verbose:
            st.subheader("📜 JSON Query")
            st.code(json_query, language="json")

        # STEP 3 — Generate + Retry with LLM Error Fixing
        error_msg = ""
        for attempt in range(1, MAX_RETRIES + 1):
            if verbose:
                st.markdown(f"🔁 **Attempt {attempt}**")

            code = generate_pandas_code(json_query, error_msg)
            valid, err = verify_code(code, df)

            if valid:
                break
            else:
                error_msg = f"The last code failed with error:\n{err}"
                if attempt == MAX_RETRIES:
                    st.error("❌ All code generation attempts failed.")
                    st.info(f"🧠 LLM Error Insight:\n{err}")
                    st.stop()

        # STEP 4 — Display and Run Code
        st.subheader("🐍 Final Pandas Code")
        st.code(code, language="python")
        st.subheader("📊 Output Table")
        result_df = execute_pandas_code(code, df)

        summary = summarize_result(result_df, query, summary_mode, dataset_description)

        st.subheader("🧾 Summary")
        st.write(summary)
