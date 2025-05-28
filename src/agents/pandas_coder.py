import json
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────
llm = OllamaLLM(model="qwen2.5-coder:7b")

# ─────────────────────────────────────────────────────────
pandas_prompt = PromptTemplate(
    input_variables=["json_query", "error_msg"],
    template="""
You are a highly precise Python assistant that generates valid and executable Pandas code from a JSON query.

You will also improve the code based on any previous error explanations.

---

## JSON Query (User Intent):
{json_query}

## Error Message (From Validator):
{error_msg}

---

### Your Task:

1. Carefully read the **error message** to understand what failed last time:
   - If the issue involves column names, fix the incorrect name.
   - If it's a logical contradiction, adjust the condition logic.
   - If the formula syntax was incorrect, use standard Python syntax and fix the structure.
   - If the error mentions operator misuse or missing parenthesis, correct those directly.

2. Rebuild the code from scratch using the corrected logic.
   - Integrate ONLY what the error message describes — don’t add extra fixes or logic unless clearly required.

---

### Code Construction Rules:

1. Use `df` as the name of the DataFrame.

2. For each condition in the JSON:
   - If `"field"` is used:
     → Use the format: `df["<Field Name>"] <operator> <value>`
   - If `"formula"` is used:
     → Parse it and translate to Pythonic syntax:
       - Replace columns with `df["..."]`
       - Math symbols:
         - `×` → `*`
         - `−` or `-` → `-`
         - `÷` → `/`
         - `+` → `+`

3. Combine multiple conditions using:
   - `&` for AND
   - `|` for OR
   - Wrap each condition in parentheses.

4. Apply selection if `select_fields` exists:
   → Use `.loc[condition, [list of columns]]`

5. DO NOT:
   - Return any markdown, explanations, or notes
   - Return more than one code block
   - Use undefined variables or missing columns

6. NEVER create a new DataFrame with sample data. You must use the existing `df` provided.

7. Assume `df` is already loaded with the correct dataset. Do not redefine or initialize `df`.

8. DO NOT use any hardcoded or fake sample data.

---

### Example Fixes Based on Errors:

- If the error says: "Column 'XYZ' not found"
  → Replace with the correct column name that exists in the dataset. Check for typos or case mismatches (e.g., use `"Type"` if that's the actual name).

- If the error says: "Invalid comparison between string and number"
  → Wrap string values in quotes (e.g., `"value"`) to ensure the comparison is between compatible types.

- If the error says: "Logical contradiction"
  → Avoid impossible conditions such as checking if a value is both greater than A and less than B when A > B. Adjust the logic to match a valid range or condition.

- If the error says: "Missing closing bracket" or "unmatched parentheses"
  → Make sure that all parentheses and square brackets are properly closed, especially in `.loc[...]` or multi-condition filters.

---

Output Requirement:
- Your final line MUST assign the filtered result to a variable named `result`.
- Example: `result = df.loc[...]` or `result = df[...]`
- You may define intermediate variables (like `condition = ...`) above it if needed.
- Do NOT return code without the assignment to `result`.
- Do NOT include markdown, explanations, or comments — only valid Python code.
"""
)

pandas_coder_chain = pandas_prompt | llm

# ─────────────────────────────────────────────────────────
def generate_pandas_code(json_query: str, error_msg: str = "") -> str:
    """Generates Pandas code from JSON and (optionally) retries based on LLM error."""
    return pandas_coder_chain.invoke({
        "json_query": json_query,
        "error_msg": error_msg
    }).strip()
