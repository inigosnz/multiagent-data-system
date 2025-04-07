from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import re
import pandas as pd

llm = OllamaLLM(model="deepseek-r1:7b") 

summary_prompt = PromptTemplate(
    input_variables=["table", "query", "style", "description"],
    template="""
You are a highly skilled data analyst with a deep understanding of numerical patterns, units, and statistics.

Your task is to summarize a filtered DataFrame below. Use only the columns and meanings provided in the dataset description. Generate a natural language summary tailored to the user's query and desired style.

---

User Query:
"{query}"

Dataset Description (column meanings and units):
{description}

Filtered Dataset Preview:
{table}

Summary Style: {style}

---

Guidelines:

- Use the dataset description to interpret column meanings — do not guess.
- Do not hallucinate information outside of the data.
- Use only values and patterns you can see in the dataset.
- If there is a mismatch between the query and the table, you may note it.

---

Summary Style Guide:

- "Short":
  • Provide a quick, high-level summary in 2–3 sentences.

- "Detailed":
  • Reference numeric trends, ranges, or relationships across multiple columns.

- "Stats-heavy":
  • Provide statistical measures (mean, min, max, count, etc.) across relevant columns.

---

Output:
Write a clear, plain English summary. Do not include any code, markdown, or headings.
"""
)


summarizer_chain = summary_prompt | llm

def summarize_result(df: pd.DataFrame, query: str, style: str, dataset_description: str) -> str:
    preview = df.to_markdown(index=False)
    raw_output = summarizer_chain.invoke({
        "query": query,
        "style": style,
        "table": preview,
        "description": dataset_description
    }).strip()

    # Remove <think> ... </think> block if it exists
    cleaned_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

    return cleaned_output


