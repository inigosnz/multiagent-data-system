import json
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────
llm = OllamaLLM(model="deepseek-r1:7b")

# ─────────────────────────────────────────────────────────
# Extract all column names from dataset description
def extract_columns(dataset_description):
    """Extracts column names from dataset description."""
    lines = dataset_description.split("\n")
    columns = []

    for line in lines:
        if "**" in line:
            col_name = line.split("**")[1].split("**")[0]
            columns.append(col_name)

    return columns

# ─────────────────────────────────────────────────────────
json_prompt = PromptTemplate(
    input_variables=["conditions", "dataset"],
    template="""
You must return a valid JSON object in the exact format described below.

---

SYSTEM-WIDE REQUIREMENT:
- Include **all actual dataset columns** in `select_fields`

---

Forbidden:
- DO NOT write SQL code.
- DO NOT return SELECT statements, WHERE clauses, or database queries.
- DO NOT use SQL operators like `LIKE`, `BETWEEN`, or functions like `ISNULL`.
- Create new column names or modify existing ones, you must use the exact names from the dataset description.

You must return only valid JSON following the structure described below.

---

## Dataset Description:
{dataset}

## Extracted Conditions:
{conditions}

---

## JSON Output Format:
Return only this structure:
{{
  "query_type": "select",
  "select_fields": [List of dataset column names],
  "conditions": [
    {{
      "field": "[Exact Column Name]",
      "operator": "[>, <, =, ∈]",
      "value": [Value or List]
    }},
    {{
      "formula": "[Mathematical expression using column names]",
      "operator": "[>, <]",
      "value": [Value]
    }}
  ]
}}

---

RULES FOR FIELDS AND SELECT FIELDS:
- You may ONLY use column names **from the dataset description above**.
- DO NOT invent, guess, or infer any additional column names (e.g. "Power", "Load", "Voltage").
- Match column names **exactly** — including units, capitalization, and brackets.
- Use only `"field"` or `"formula"` to define each condition (never both in one).

---

GENERAL LOOKUP QUERIES:
If conditions include only a `Product ID` or `UDI` equality condition:

Then:
- Include **all actual dataset columns** in `select_fields`
- Do NOT include any failure logic, inferred formulas, or extra columns

---

Output Requirements:
- All strings in double quotes
- No markdown, explanation, or extra text
- Return ONLY the raw JSON object
"""
)

json_formatting_chain = json_prompt | llm

# ─────────────────────────────────────────────────────────
# Clean JSON block
def extract_json_block(text):
    """Extract the first valid JSON block from a string (inside ``` or not)."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()

# ─────────────────────────────────────────────────────────
def format_to_json(conditions, dataset_description):
    """Converts extracted conditions into JSON format and adjusts select_fields intelligently."""

    formatted_json = json_formatting_chain.invoke({
        "conditions": conditions,
        "dataset": dataset_description
    }).strip()

    cleaned_json = extract_json_block(formatted_json)

    try:
        parsed_json = json.loads(cleaned_json)
        return json.dumps(parsed_json, indent=2)

    except json.JSONDecodeError as e:
        print("\n❌ Invalid JSON while formatting:", e)
        return cleaned_json
