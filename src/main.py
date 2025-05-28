import streamlit as st
import pandas as pd

from agents.query_extractor import extract_conditions
from agents.json_formatter import format_to_json
from agents.pandas_coder import generate_pandas_code
from agents.code_verifier import verify_code
from agents.code_executor import execute_pandas_code
from agents.question_solver import chat_about_data
from agents.analyst_agent import analyze_data

MAX_RETRIES = 10

# App setup
st.set_page_config(page_title="Multi-Agent Data System", page_icon="📊", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "trigger_query" not in st.session_state:
    st.session_state.trigger_query = False

if "show_df_on_chat" not in st.session_state:
    st.session_state.show_df_on_chat = False

# Sidebar
st.sidebar.header("🧾 Upload Your Files")
csv = st.sidebar.file_uploader("📁 CSV File", type=["csv"])
dataset_description_file = st.sidebar.file_uploader("📝 Description File", type=["txt"])

st.sidebar.markdown("---")

st.sidebar.header("🔧 Settings")
verbose = st.sidebar.checkbox("🔊 Verbose mode", value=True)

debug_mode = st.sidebar.checkbox("🐞 Enable Debug Mode", value=True)

st.sidebar.markdown("---")

st.sidebar.header("🧹 Cleaning section")
if st.sidebar.button("📃 Clear Screen"):
    for key in list(st.session_state.keys()):
        if key not in ["chat_history"]:
            del st.session_state[key]
    st.rerun()

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared!")
    st.rerun()


st.sidebar.markdown("---")
st.sidebar.header("📬 Contact")
st.sidebar.markdown(
    """
    <div style='text-align: center;'>
        <a href="https://github.com/inigosnz" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-inigosnz-black?style=for-the-badge&logo=github" alt="GitHub Badge">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)



# Layout
app_col, chat_col = st.columns([1, 1])

with app_col:
    st.title("🤖 Multi-Agent Pandas Query System")

    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
### 📘 Welcome to the Multi-Agent Pandas Query System

This app lets you explore your data using **natural language**, powered by intelligent agents that:
- Extract logic from your questions
- Generate and validate real Python code
- Execute data filters and transformations
- Explain results like a data analyst — using metrics, correlations, and summaries

---

### 🚀 Step-by-Step Guide

1. **Upload your files**
   - **CSV file**: This is the raw dataset you'll explore.
   - **Dataset description**: A plain `.txt` file that defines column names, meanings, and units.

2. **Enter your query**
   - Use **natural language** to ask for insights or filters.
   - Examples:
     - *"Show only samples where temperature is above 85°C."*
     - *"Is there a correlation between torque and speed?"*

3. **Click “Run”**
   - The system will:
     - Parse your intent
     - Generate valid Pandas code
     - Validate and fix errors if needed
     - Show the filtered result

4. **Use the Chat for follow-up questions**
   - Ask questions like:
     - *“Why are these values so low?”*
     - *“What happens to pressure when speed increases?”*
   - The assistant will:
     - Use statistical summaries
     - Compare against full dataset
     - Speak like a real analyst

---

Any questions? Feel free to reach out  

"""
)

    query = st.text_input("🔍 Enter your query:", key="main_query")

    if st.button("Run"):
        if not csv or not dataset_description_file:
            st.error("❌ Please upload both a CSV file and a dataset description file before running the query.")
        else:
            st.session_state.trigger_query = True


    if st.session_state.get("trigger_query"):
        with st.spinner("Processing your query..."):
            df = pd.read_csv(csv)
            dataset_description = dataset_description_file.read().decode("utf-8")

            extracted = extract_conditions(query, dataset_description)
            if verbose:
                with st.expander("🧠 Extracted Conditions"):
                    st.code(extracted)

            json_query = format_to_json(extracted, dataset_description)
            if verbose:
                with st.expander("🧾 JSON Query"):
                    st.code(json_query, language="json")

            error_msg = ""
            for attempt in range(1, MAX_RETRIES + 1):
                if verbose:
                    st.info(f"🔄 Attempt {attempt}")

                code = generate_pandas_code(json_query, error_msg)
                valid, result_or_error = verify_code(code, df, df, dataset_description)

                if valid:
                    break
                else:
                    if isinstance(result_or_error, str) and verbose:
                        with st.expander("❌ LLM Error Message"):
                            st.info(result_or_error)

                    error_msg = f"The last code failed with error:\n{result_or_error}"

                    if attempt == MAX_RETRIES:
                        st.error("❌ All code generation attempts failed.")
                        st.info(f"🧠 LLM Error Insight:\n{result_or_error}")
                        st.stop()

            if verbose:
                with st.expander("🐍 Final Pandas Code"):
                    st.code(code, language="python")

            result_df = execute_pandas_code(code, df)

            st.session_state.df = df
            st.session_state.result_df = result_df
            st.session_state.dataset_description = dataset_description
            st.session_state.code = code
            st.session_state.trigger_query = False

    if (
        "result_df" in st.session_state and 
        (st.session_state.get("show_df_on_chat") or not st.session_state.get("trigger_query"))
    ):
        st.subheader("📊 Output Table")
        st.dataframe(st.session_state.result_df, use_container_width=True)
        st.caption(f"✅ {len(st.session_state.result_df)} rows × {len(st.session_state.result_df.columns)} columns shown")
        st.session_state.show_df_on_chat = False

with chat_col:
    st.subheader("💬 Chat About Your Data")

    # Group messages into pairs: (user → assistant)
    history_pairs = [
        tuple(st.session_state.chat_history[i:i+2])
        for i in range(0, len(st.session_state.chat_history) - 1, 2)
    ]

    # 🔁 Show older messages in an expander
    if len(history_pairs) > 1:
        with st.expander("🕓 Previous Messages", expanded=False):
            for user_msg, assistant_msg in history_pairs[:-1]:
                with st.chat_message("user"):
                    st.markdown(user_msg.replace("User: ", "", 1))
                with st.chat_message("assistant"):
                    st.markdown(assistant_msg.replace("Assistant: ", "", 1))

    # 🆕 Show only the latest message pair
    if history_pairs:
        user_msg, assistant_msg = history_pairs[-1]
        with st.chat_message("user"):
            st.markdown(user_msg.replace("User: ", "", 1))
        with st.chat_message("assistant"):
            st.markdown(assistant_msg.replace("Assistant: ", "", 1))


    # Chat input
    user_input = st.chat_input("Ask a follow-up question...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append(f"User: {user_input}")
        st.session_state.show_df_on_chat = True

        with st.spinner("🔎 Analyzing your question... this may take a few seconds."):
            debug_result = analyze_data(
                result_df=st.session_state.result_df,
                full_df=st.session_state.df,
                dataset_description=st.session_state.dataset_description,
                user_question=user_input
            )

            if debug_result["metrics"] is None:
                response = debug_result["error"]
            else:
                response = chat_about_data(
                    user_input=user_input,
                    metrics=debug_result["metrics"],
                    dataset_description=st.session_state.dataset_description,
                    history=st.session_state.chat_history,
                    verbose=verbose  
                )

                # 🔁 Re-run if AI said new analysis is required
                if "requires new analysis" in response.lower():
                    with st.spinner("🔁 Re-analyzing..."):
                        debug_result = analyze_data(
                            result_df=st.session_state.result_df,
                            full_df=st.session_state.df,
                            dataset_description=st.session_state.dataset_description,
                            user_question=user_input
                        )
                        if debug_result["metrics"] is None:
                            response = debug_result["error"]
                        else:
                            response = chat_about_data(
                                user_input=user_input,
                                metrics=debug_result["metrics"],
                                dataset_description=st.session_state.dataset_description,
                                history=st.session_state.chat_history,
                                verbose=verbose  
                            )

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append(f"Assistant: {response}")

        if debug_mode:
            with st.expander("🧠 Analyst Agent Debug Info", expanded=False):
                st.subheader("🧾 Generated Code")
                st.code(debug_result["code"], language="python")

                st.subheader("📊 Raw Metric Output")
                st.text(debug_result["metrics"])
