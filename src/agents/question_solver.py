from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import pandas as pd
import re

llm = OllamaLLM(model="deepseek-r1:7b")

# ─────────────────────────────────────────────────────────
# QUESTION PROMPT
# ─────────────────────────────────────────────────────────

question_solver_prompt = PromptTemplate(
    input_variables=["metrics", "user_input", "history", "description"],
    template="""
You are a helpful data assistant.

The user asked a question about a dataset. An Analyst Agent computed several metrics based on both the filtered data and the full dataset.

Your job is to explain these metrics clearly, using professional and friendly language.

---

🧑 User Question:
{user_input}

📘 Dataset Description:
{description}

📊 Computed Metrics:
{metrics}

🗃️ Conversation History:
{history}

---

🗣️ Language: Always use English for all reasoning and final answers.

---

🎯 Your Task:
- Use the metrics above to answer the user's question.
- Explain what each metric means and how the filtered and full datasets compare.
- Indicate if values increased, decreased, or remained stable.
- If changes are meaningful, mention them (e.g. % or absolute differences).
- Refer to column names exactly as shown.
- If metrics do not reveal meaningful differences, say so clearly.
- Only use the metrics provided — do not invent extra logic or analysis.
- Use short, clear, and factual paragraphs.

If the metrics do not answer the user’s question, respond:
> "The Analyst Agent did not cover this question yet. It requires new analysis."

Avoid code, markdown, or plots.
"""
)


question_chain = question_solver_prompt | llm

# ─────────────────────────────────────────────────────────
# MAIN FUNCTION 
# ─────────────────────────────────────────────────────────

def chat_about_data(
    user_input: str,
    metrics: str,
    dataset_description: str,
    history: list[str],
    verbose: bool = False 
) -> str:
    import re

    history_text = "\n".join(history)

    raw_response = question_chain.invoke({
        "user_input": user_input,
        "metrics": metrics,
        "history": history_text,
        "description": dataset_description
    }).strip()

    # Buscar bloques de razonamiento dentro de <think>...</think>
    think_blocks = re.findall(r"<think>(.*?)</think>", raw_response, re.DOTALL)

    # Resto del mensaje después del último </think>
    response_cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

    if verbose:
        reasoning = "\n\n".join(f"🔍 **AI Reasoning:**\n\n{block.strip()}" for block in think_blocks)
        answer = f"\n\n✅ **Final Answer:**\n\n{response_cleaned}" if response_cleaned else ""
        return reasoning + answer if reasoning or answer else "⚠️ No explanation provided."
    else:
        return response_cleaned or "⚠️ No final answer provided."

