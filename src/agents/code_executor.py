import pandas as pd
import streamlit as st
import re

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
def execute_pandas_code(code: str, df: pd.DataFrame):
    local_scope = {"df": df.copy()}
    cleaned_code = clean_code(code)

    if not cleaned_code.startswith("result ="):
        cleaned_code = f"result = {cleaned_code}"

    try:
        exec(cleaned_code, {}, local_scope)
        result = local_scope.get("result")

        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
            st.caption(f"✅ Displayed {len(result)} rows × {len(result.columns)} columns")
            return result  
        else:
            st.write(result)
            return None  

    except Exception as e:
        st.error(f"❌ Execution failed: {e}")
        return None