import pandas as pd
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from agents.code_verifier import verify_code

# ─────────────────────────────────────────────
llm_coder = OllamaLLM(model="qwen2.5-coder:7b")
llm_writer = OllamaLLM(model="deepseek-r1:7b")

# ─────────────────────────────────────────────
def clean_code(code: str) -> str:
    if code.startswith("```"):
        code = re.sub(r"```[a-zA-Z]*", "", code)
        code = code.replace("```", "")
    return code.strip()

def format_metrics_output(result) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.to_string()

    if isinstance(result, dict):
        output = []
        for key, value in result.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                output.append(f"{key}:\n{value.to_string()}")
            else:
                output.append(f"{key}: {value}")
        return "\n".join(output)
    return str(result)

# ─────────────────────────────────────────────
# CODE GENERATION WITH FULL DATAFRAME AWARENESS
code_prompt = PromptTemplate(
    input_variables=["question", "columns", "error"],
    template="""
You are a Python data analyst tasked with generating pandas code to extract the most relevant metrics for answering the user's question.

You have access to two DataFrames:
- `df`: the filtered dataset based on the user's query
- `full_df`: the complete dataset for reference

---

🧠 USER QUESTION:
{question}

📊 COLUMN NAMES IN THE DATASET:
{columns}

❌ PREVIOUS ERROR (if any):
{error}

---

🛠️ INSTRUCTIONS:
- Analyze BOTH `df` (filtered) AND `full_df` (original).
- Compute metrics for both datasets side by side:
  - Averages, correlations, group comparisons, failure rates, etc.
- Clearly label metrics from each dataset:
  - Example: `avg_speed_filtered`, `avg_speed_full`, `diff_avg_speed`
- Compute differences and highlight patterns:
  - Differences (`-`), ratios (`/`), percent changes (e.g., `(A - B)/B * 100`)
- Store everything in a dictionary called `result`
- Do NOT use plots or markdown
- DO NOT invent or abbreviate column names — use them **exactly** as provided
- DO NOT rename categories (e.g. keep `"Product Type"` as is — don't map L/M/H)

✅ EXAMPLE OUTPUT:
result = {{
    "avg_speed_filtered": df["Rotational speed [rpm]"].mean(),
    "avg_speed_full": full_df["Rotational speed [rpm]"].mean(),
    "diff_avg_speed": df["Rotational speed [rpm]"].mean() - full_df["Rotational speed [rpm]"].mean(),
    "failure_rate_filtered": df["Machine failure"].mean(),
    "failure_rate_full": full_df["Machine failure"].mean()
}}

Return ONLY valid Python code (no markdown).
"""
)


code_chain = code_prompt | llm_coder

# ─────────────────────────────────────────────
# EXPLANATION WRITER
writer_prompt = PromptTemplate(
    input_variables=["metrics", "description", "question"],
    template="""
You are a data analyst writing a clear, factual summary of the extracted metrics in response to the user's question.

The analysis includes metrics from two datasets:
- `df`: the filtered data
- `full_df`: the complete original dataset

---

🧠 USER QUESTION:
{question}

📘 DATASET DESCRIPTION:
{description}

📊 METRICS (already computed in Python):
{metrics}

---

✅ GUIDELINES:
- Only use the metrics explicitly shown above
- Explain each metric’s meaning, and compare filtered vs full dataset:
  - Indicate if values increased, decreased, or stayed the same
  - Mention % changes or absolute differences where possible
- Refer to column names exactly as written in the dataset
- Do NOT:
  - Suggest additional analysis or techniques (e.g. ANOVA, regression)
  - Mention visualization methods
  - Invent thresholds, logic, or statistical tests
  - Add interpretations not directly supported by the metrics
- If no meaningful difference is observed, state that clearly
- Use short, factual paragraphs (no markdown or code)
"""
)



writer_chain = writer_prompt | llm_writer

# ─────────────────────────────────────────────
# MAIN FUNCTION TO COMBINE EVERYTHING
def summarize_result_with_context(
    result_df: pd.DataFrame,
    full_df: pd.DataFrame,
    dataset_description: str,
    user_question: str,
    max_retries: int = 10
) -> dict:
    columns = "\n".join(f"- {col}" for col in full_df.columns)
    error = ""
    code = ""
    success = False
    raw_result = None

    for attempt in range(1, max_retries + 1):
        code = code_chain.invoke({
            "question": user_question,
            "columns": columns,
            "error": error or "None"
        }).strip()

        code = clean_code(code)

        success, raw_result = verify_code(code, df=result_df, full_df=full_df , dataset_description=dataset_description)

        if success:
            break
        error = raw_result

    if not success:
        return {
            "code": code,
            "metrics": raw_result,
            "explanation": f"❌ Failed to execute code after {max_retries} attempts.\n\n{raw_result}"
        }

    formatted_metrics = format_metrics_output(raw_result)

    explanation = writer_chain.invoke({
        "metrics": formatted_metrics,
        "description": dataset_description,
        "question": user_question
    }).strip()

    return {
        "code": code,
        "metrics": formatted_metrics,
        "explanation": explanation
    }
