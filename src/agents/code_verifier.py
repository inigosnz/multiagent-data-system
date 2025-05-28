import pandas as pd
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────
llm = OllamaLLM(model="qwen2.5-coder:7b")

# ─────────────────────────────────────────────────────────
# Clean Code 
def clean_code(code: str) -> str:
    """Remove markdown and clean up code."""
    if code.startswith("```"):
        code = re.sub(r"```[a-zA-Z]*", "", code)
        code = code.replace("```", "")
    return code.strip()

# ─────────────────────────────────────────────────────────
error_explainer_prompt = PromptTemplate(
    input_variables=["code", "error", "description"],
    template="""
You are an expert Python assistant specializing in Pandas DataFrame filtering.

You must help a developer understand why their code failed and what exactly to fix — without returning code.

Use the provided dataset description to verify column names, units, and meaning. Do not guess column names — match them exactly as described.

---

## Code to Analyze:
{code}

## Error Message:
"{error}"

## Dataset Description (column names and definitions):
{description}

---

### Your Task:

1. **Diagnose the root cause**:
   - Is it a syntax mistake?
   - A logic contradiction?
   - A column name or operator issue?
   - A data type mismatch?

2. **Point out the faulty line or expression** that triggered the error.
   - Quote it explicitly if possible.
   - Reference the relevant line or fragment clearly.

3. **Explain the issue** in plain, clear English.

4. **Suggest a concrete fix**:
   - Specify which operator or expression is wrong.
   - Suggest the exact value, column name, or part of the formula to change.
   - Refer directly to line logic or math as needed.
   - Do NOT return code — describe what to do precisely.

---

### Examples of Ideal Explanations:

**Example – Column not found:**
> The condition `df["Incorrect Column Name"]` failed because the dataset contains the column `"Torque [Nm]"`. Pandas is case-sensitive — use the exact column name as described.

**Example – Data type mismatch:**
> You're comparing a column that contains strings to a number: `123`. This causes a type error — wrap the value in quotes if it's a string.

**Example – Logical contradiction:**
> The condition `(value > 9000) & (value < 3500)` is logically impossible — revise the logic.

---

Precision Matters:
- Refer to the **exact columns** described in the dataset.
- Be direct and clear — no code blocks or markdown.
- ONLY return the explanation and what exactly needs to be fixed.
"""
)

error_explainer_chain = error_explainer_prompt | llm

# ─────────────────────────────────────────────────────────
def explain_error_with_llm(code: str, error: str, description: str) -> str: 
    return error_explainer_chain.invoke({
        "code": code,
        "error": error,
        "description": description
    }).strip()

def verify_code(code: str, df: pd.DataFrame, full_df: pd.DataFrame, dataset_description: str) -> tuple[bool, object]:
    """
    Executes code safely. Returns (success: bool, result or error explanation).
    Uses LLM to explain if code fails. Supports both filtered and full datasets.
    """
    local_scope = {
        "df": df.copy(),             
        "full_df": full_df.copy(),   
    }

    try:
        code = clean_code(code)

        # Check column existence
        used_cols = re.findall(r'df\["([^"]+)"\]', code) + re.findall(r'full_df\["([^"]+)"\]', code)
        all_cols = set(df.columns).union(set(full_df.columns))
        missing_cols = [col for col in used_cols if col not in all_cols]

        if missing_cols:
            raise KeyError(f"The following columns do not exist in the dataset: {missing_cols}")

        # Execute the code
        exec(code, {}, local_scope)

        # Verify that 'result' exists and is a valid object
        if "result" not in local_scope:
            raise NameError("No variable named `result` found in the code.")

        result = local_scope["result"]

        # Optionally validate type here (e.g., must be dict or DataFrame)
        if not isinstance(result, (dict, pd.DataFrame, pd.Series)):
            raise TypeError(f"The `result` variable is not a valid type. Found: {type(result)}")

        return True, result

    except Exception as e:
        error_type = type(e).__name__
        full_error = f"{error_type}: {str(e)}"

        explanation = explain_error_with_llm(code, full_error, dataset_description)
        return False, explanation

