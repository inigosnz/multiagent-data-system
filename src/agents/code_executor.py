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
        code = code.replace("```", "") 

    code_lines = [line for line in code.splitlines() if line.strip()]
    return "\n".join(code_lines).strip()

# ─────────────────────────────────────────────────────────
# Execute Code 
def execute_pandas_code(code: str, df: pd.DataFrame) -> pd.DataFrame:
    local_scope = {"df": df.copy()}
    cleaned_code = clean_code(code)

    try:
        exec(cleaned_code, {}, local_scope)
        result = local_scope.get("result")
        return result if isinstance(result, pd.DataFrame) else None

    except Exception as e:
        st.error(f"❌ Execution failed: {e}")
        return None

