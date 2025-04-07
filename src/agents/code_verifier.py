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
    input_variables=["code", "error"],
    template="""
You are an expert Python assistant specializing in Pandas DataFrame filtering.

You must help a developer understand why their code failed and what exactly to fix — without returning code.

---

## Code to Analyze:
{code}

## Error Message:
"{error}"

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

**Example 1 – SyntaxError (missing parenthesis):**
> The error occurred in the expression `((df["Some Column"] > 200` because the closing parenthesis is missing. Python cannot evaluate the full condition until it’s properly closed. Add the missing `)` after the comparison to fix the structure.

**Example 2 – Logical contradiction:**
> In your condition `(value > 9000) & (value < 3500)`, you're checking for a number to be both greater than 9000 and less than 3500 at the same time, which is not logically possible. Choose either the upper or lower bound depending on what you're filtering for.

**Example 3 – Wrong operator:**
> The snippet `df["Some ID"] = "XYZ123"` uses a single equals sign (`=`), which is invalid for comparisons. Use double equals `==` to check for equality in Pandas filter expressions.

**Example 4 – Column not found:**
> The condition `df["Incorrect Column Name"]` failed because the dataset contains a different column, such as `correct column name` (with a lowercase letter or different spacing). Pandas is case-sensitive — you must use the exact column name from the dataset.

**Example 5 – Data type mismatch:**
> You’re comparing a column that contains strings to a number: `123`. String comparisons need to be made using quotes, like `"123"`. Wrap the value in quotes to avoid the mismatch.

**Example 6 – Formula symbol issue:**
> The expression `Column A × Column B` uses a multiplication symbol (`×`) that is not valid in Python. Pandas requires the asterisk `*` for multiplication. Replace non-code math symbols with proper Python syntax.

---

Precision Matters:
- Be surgical and direct.
- Point out the **exact fix** to apply, even if it's as small as switching `or` to `|`.

DO NOT:
- Return any Python or Pandas code.
- Use markdown or code blocks.

ONLY return the explanation and what exactly needs to be fixed.
"""
)

error_explainer_chain = error_explainer_prompt | llm

# ─────────────────────────────────────────────────────────
def explain_error_with_llm(code: str, error: str) -> str: 
    return error_explainer_chain.invoke({"code": code, "error": error}).strip()

def verify_code(code: str, df: pd.DataFrame) -> tuple[bool, str]:
    """
    Executes code safely. Returns success flag and either an empty string or an LLM-based explanation of the error.
    """
    local_scope = {"df": df.copy()}
    try:
        code = clean_code(code)
        if not code.startswith("result ="):
            code = f"result = {code}"
        exec(code, {}, local_scope)
        return True, ""
    except Exception as e:
        return False, explain_error_with_llm(code, str(e))
