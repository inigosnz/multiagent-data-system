from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import pandas as pd
import re

llm = OllamaLLM(model="deepseek-r1:7b")

# ─────────────────────────────────────────────────────────
# QUESTION PROMPT
# ─────────────────────────────────────────────────────────

question_solver_prompt = PromptTemplate(
    input_variables=["analysis", "user_input", "history"],
    template="""
You are a helpful data assistant.

The user asked a question, and the Analyst Agent has already analyzed the data and written an explanation.

---

🧑 User Question:
{user_input}

📈 Analyst Agent Explanation:
{analysis}

🗃️ Conversation History:
{history}

---

🎯 Your Task:
- Clearly explain the Analyst Agent’s answer using professional language.
- Reference the history only if it helps clarify what the user meant or if this is a follow-up.
- Do not add new analysis or assumptions.
- If the Analyst Agent's explanation already answers the question, summarize that clearly.
- If the Analyst Agent did NOT actually answer the user’s question or the explanation is unrelated, say:
  > "The Analyst Agent did not cover this question yet. A new analysis may be needed."

Keep your tone friendly and concise.
"""
)


question_chain = question_solver_prompt | llm

# ─────────────────────────────────────────────────────────
# MAIN FUNCTION 
# ─────────────────────────────────────────────────────────

def chat_about_data(
    user_input: str,
    analysis: str,
    history: list[str]
) -> str:
    history_text = "\n".join(history)

    # If the analyst already requested a new analysis — no need to rephrase
    if "requires new analysis" in analysis.lower():
        return "This question requires new analysis. Let me check that for you..."

    # Otherwise, rephrase the analyst's explanation
    raw_response = question_chain.invoke({
        "user_input": user_input,
        "analysis": analysis,
        "history": history_text,
    }).strip()

    return re.sub(r"<think>", "**AI thinking...**\n\n", raw_response).replace("</think>", "")
