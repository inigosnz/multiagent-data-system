from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import os

# ─────────────────────────────────────────────────────────
llm = OllamaLLM(model="deepseek-r1:7b")

# ─────────────────────────────────────────────────────────
# Load dataset 
data_folder = os.path.join(os.path.dirname(__file__), "../data")
dataset_file_path = os.path.abspath(os.path.join(data_folder, "dataset_description.txt"))

with open(dataset_file_path, "r") as f:
    dataset_description = f.read()

# ─────────────────────────────────────────────────────────
from langchain.prompts import PromptTemplate

extraction_prompt = PromptTemplate(
    input_variables=["query", "dataset"],
    template="""
You are an expert assistant for converting natural language queries into structured filtering conditions for a dataset.

Return only human-readable filter instructions in plain text. Do not return any code, SQL, JSON, or markdown.

---

GLOBAL RULES:

1. Do not return Python, Pandas, SQL, JSON, or markdown syntax.
2. Use plain text in this format:
   - Field: ...
   - Operator: ...
   - Value: ...
   Or, for calculations:
   - Formula: ...
   - Operator: ...
   - Value: ...
3. Only extract what the user asks for. Do not assume or infer anything else.

---

Dataset Description:
{dataset}

User Query:
"{query}"

---

EXTRACTION RULES:

1. General Lookups:
If the query asks for the "status", "record", "state" or a synonym of these of a machine or product using a Product ID or UDI:
- Extract a single condition for the ID or UDI.

Example Query:  
"What is the state of product M14860?"  
Output:  
Field: Product ID  
Operator: =  
Value: M14860

Example Query:  
"Show the records for the product with UDI 12?"  
Output:  
Field: UDI
Operator: =  
Value: 12

2. Column Name Matching:
- Match column names exactly as shown in the dataset description.
- Do not rename or modify columns.
- Include brackets and units exactly (e.g., "Torque [Nm]").

3. Field vs Formula:
- Use Field when the condition involves a single column.
- Use Formula when the condition includes math between columns:
  Examples:
  - Formula: Tool wear [min] × Torque [Nm]
  - Formula: Process temperature [K] - Air temperature [K]
- Do not use Python syntax (`df[...]`, `*`, etc.). Use readable math symbols like ×, /, -, +.

String Operators Allowed (for fields like Product ID):
- startswith
- endswith
- contains

Example:
Field: Product ID  
Operator: endswith  
Value: 60

4. Product References:
- For references to product quality (L, M, H), use:
  Field: Type  
  Operator: =  
  Value: L / M / H

- For specific product identifiers, use:
  Field: Product ID  
  Operator: =  
  Value: [value]

Do not use operator ∈ unless a list is explicitly stated.
Important if asked about product type, refer to the Type column.

5. Failure Mode Rules:
If the query references failure modes (TWF, HDF, PWF, OSF), apply:

- TWF:
  Field: Tool wear [min], Operator: >, Value: 200

- HDF:
  Formula: Process temperature [K] - Air temperature [K], Operator: <, Value: 8.6  
  Field: Rotational speed [rpm], Operator: <, Value: 1380

- PWF:
  Formula: Torque [Nm] × Rotational speed [rpm] × (2π / 60), Operator: < or >, Value: 3500 or 9000

- OSF:
  Formula: Tool wear [min] × Torque [Nm], Operator: >, Value: 12000 (or higher, based on product Type)

6. Machine Failure Field:
If the query uses terms like "machine failure", "failed", "no failure":
- Use this column only: Machine failure
- Apply:
  - Field: Machine failure
  - Operator: =
  - Value: 1 (if failure)
  - Value: 0 (if no failure)

Do not calculate or infer machine failure based on TWF, PWF, HDF, or OSF if the query already mentions "machine failure". This rule overrides failure logic when used.

---

Output Format (Plain Text Only):

Use only the following structure:

- Field: [Exact Column Name]  
- Operator: [=, >, <, ∈, contains, etc.]  
- Value: [Exact Value or List]

Or:

- Formula: [Column1 × Column2 or Column1 - Column2]  
- Operator: [>, <, =]  
- Value: [Numeric Value]

Do not include any code, markdown, or JSON.
"""
)


query_extraction_chain = extraction_prompt | llm

# ─────────────────────────────────────────────────────────

def extract_conditions(query):
    return query_extraction_chain.invoke({
        "query": query,
        "dataset": dataset_description
    }).strip()
