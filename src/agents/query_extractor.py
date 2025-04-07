from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

# ─────────────────────────────────────────────────────────
llm = OllamaLLM(model="deepseek-r1:7b")

# ─────────────────────────────────────────────────────────
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

1. Column Name Matching:
- Match column names exactly as shown in the dataset description.
- Do not rename or modify columns.

2. Field vs Formula:
- Use Field when the condition involves a single column.
- Use Formula when the condition includes math between columns:
  Examples:
  - Formula: Column A × Column B
  - Formula: Column C - Column D
- Do not use Python syntax (`df[...]`, `*`, etc.). Use readable math symbols like ×, /, -, +.

3. String Operators:
If the query filters by part of a string (e.g., names, IDs, labels), you can use:
- Operator: startswith
- Operator: endswith
- Operator: contains

Example:
Field: Product Code  
Operator: contains  
Value: "X12"

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
def extract_conditions(query: str, dataset_description: str) -> str:
    return query_extraction_chain.invoke({
        "query": query,
        "dataset": dataset_description
    }).strip()
