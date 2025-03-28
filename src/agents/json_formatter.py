import json
import os
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────
llm = OllamaLLM(model="deepseek-r1:7b")

# ─────────────────────────────────────────────────────────
# Load dataset 
data_folder = os.path.join(os.path.dirname(__file__), "../data")
dataset_file_path = os.path.abspath(os.path.join(data_folder, "dataset_description.txt"))

with open(dataset_file_path, "r") as f:
    dataset_description = f.read()

# ─────────────────────────────────────────────────────────
# Extract all column names from dataset description
def extract_columns(dataset_description):
    """Extracts column names from dataset description."""
    lines = dataset_description.split("\n")
    columns = []
    
    for line in lines:
        if "**" in line:  # Detect column definitions
            col_name = line.split("**")[1].split("**")[0]  # Extract column name
            columns.append(col_name)
    
    return columns

ALL_COLUMNS = extract_columns(dataset_description)

# ─────────────────────────────────────────────────────────
json_prompt = PromptTemplate(
    input_variables=["conditions", "dataset"],
    template="""
You must return a valid JSON object in the exact format described below.

---

SYSTEM-WIDE REQUIREMENT:
- Either "Product ID" or "UDI" **must be included** in the `select_fields` array, even if no filtering condition uses them.
  - If both are missing from the user query, include "Product ID" by default.
  - If one of them appears in a condition, ensure it's **also** present in `select_fields`.

---

Forbidden:
- DO NOT write SQL code.
- DO NOT return SELECT statements, WHERE clauses, or database queries.
- DO NOT use SQL operators like `LIKE`, `BETWEEN`, or functions like `ISNULL`.

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
    # Look for a JSON block inside triple backticks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: try to find the first standalone JSON-looking block
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()

# ─────────────────────────────────────────────────────────
# Format to JSON
def format_to_json(conditions):
    """Converts extracted conditions into JSON format and adjusts select_fields intelligently."""

    formatted_json = json_formatting_chain.invoke({
        "conditions": conditions,
        "dataset": dataset_description
    }).strip()

    cleaned_json = extract_json_block(formatted_json)

    try:
        parsed_json = json.loads(cleaned_json)

        # Fallback if select_fields is missing
        if "select_fields" not in parsed_json:
            parsed_json["select_fields"] = []

        # If only 'Product ID' is present, treat as full record lookup → return all columns
        if parsed_json["select_fields"] == ["Product ID"]:
            parsed_json["select_fields"] = ALL_COLUMNS

        # Otherwise, ensure 'Product ID' is present in select_fields
        elif "Product ID" not in parsed_json["select_fields"]:
            parsed_json["select_fields"].append("Product ID")

        return json.dumps(parsed_json, indent=2)

    except json.JSONDecodeError as e:
        print("\n❌ Invalid JSON while formatting:", e)
        return cleaned_json

