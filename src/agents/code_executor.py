import os
import pandas as pd
import streamlit as st
import re

# ─────────────────────────────────────────────────────────
# Load the dataset
data_folder = os.path.join(os.path.dirname(__file__), "../data")
csv_file_path = os.path.abspath(os.path.join(data_folder, "IOT.csv"))
df = pd.read_csv(csv_file_path)

# ─────────────────────────────────────────────────────────
# Clean Code 
def clean_code(code: str) -> str:
    """Sanitize code block from LLM (e.g. remove markdown like ```python)."""
    code = code.strip()

    
    if code.startswith("```"):
        code = re.sub(r"```[a-zA-Z]*", "", code)
        code = code.replace("```", "")  # remove trailing backticks

    code_lines = [line for line in code.splitlines() if line.strip()]
    return "\n".join(code_lines).strip()

# ─────────────────────────────────────────────────────────
# Execute Code 
def execute_pandas_code(code: str):
    """
    Cleans and executes LLM-generated Pandas code and displays the result in Streamlit.
    """
    local_scope = {"df": df.copy()}

    # Clean code
    cleaned_code = clean_code(code)

    # Ensure result assignment
    if not cleaned_code.startswith("result ="):
        cleaned_code = f"result = {cleaned_code}"

    # Display the cleaned code
    st.code(cleaned_code, language="python")

    try:
        exec(cleaned_code, {}, local_scope)
        result = local_scope.get("result")

        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
            st.caption(f"✅ Displayed {len(result)} rows × {len(result.columns)} columns")
        else:
            st.write(result)

        return "✅ Code executed successfully."

    except Exception as e:
        st.error(f"❌ Execution failed: {e}")
        return f"❌ Error: {e}"
