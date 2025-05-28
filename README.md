# 🤖 Multi-Agent System for Conversational Data Analysis

This project implements a modular **multi-agent architecture** for natural language interaction with structured datasets — enabling users to analyze, explore, and reason about data **without writing a single line of code**.

Built with **Streamlit**, **Pandas**, and **locally hosted LLMs (via Ollama)**, the system allows users to upload their own datasets and descriptions, ask questions in plain English, and receive **code-backed, statistically grounded answers** in a conversational interface.

---

## 🧠 What It Does

Users can interact with their dataset using chat-like queries such as:

> “Does Type L lead to higher process temperature?”  
> “Show me all samples with torque above 50 Nm and tool wear below 100.”

The system breaks down and solves this request through a team of specialized LLM agents:

- Analyst Agent – Generates Python code from structured logic, executes it, extracts statistical metrics, and explains the analysis.
- Question Solver Agent – Interprets the analysis, reasons through it, and gives a final answer.
- Query Extraction Agent – Understands natural language and converts it into logical conditions using dataset context.
- JSON Formatter Agent – Converts the logic into standardized JSON.
- Pandas Coder Agent – Generates executable Pandas code from the structured query.
- Code Verifier Agent – Validates code and retries with improvements based on LLM insights.
- Code Execution Agent – Executes safe Pandas code and returns filtered results.

---

## 💡 Core Features

✅ Conversational interface powered by intelligent agents  
✅ Natural language to query → reasoning → final answer  
✅ Visual debug of generated code, metrics, and logic  
✅ Iterative retries when code generation fails  
✅ Supports verbose reasoning mode toggle  
✅ Locally hosted, privacy-preserving LLMs (via Ollama)  
✅ Streamlit app with dual-pane chat and data view  
✅ Modular agent logic for extension (SQL, charting, etc.)

---

## 🗂️ Project Structure

```bash
multiagent-data-system/
│
├── src/
│   ├── agents/
│   │   ├── analyst_agent.py
│   │   ├── question_solver.py
│   │   ├── code_executor.py
│   │   ├── code_verifier.py
│   │   ├── json_formatter.py
│   │   ├── pandas_coder.py
│   │   └── query_extractor.py
│   ├── main.py
│
├── requirements.txt
└── README.md
```

🚀 How to Run
1. Install the dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the app:
```bash
streamlit run src/main.py
```

✅ Make sure Ollama is installed and running with required models: deepseek-r1:7b and qwen2.5-coder:7b.

⚙️ If you want to use different LLMs, you'll need to update the model names directly in each agent file (/agents/*.py) based on what your machine can run locally.

🌐 If you prefer to use external APIs (e.g., OpenAI), simply replace the llm = Ollama(...) definition in each agent file with the API-specific wrapper (e.g., llm = OpenAI(model="gpt-4", api_key="...")).


📌 Example Use Cases
Exploratory data analysis in industrial/IoT settings

Root cause analysis from failure logs

KPI evaluation and hypothesis testing

Natural language dashboards

Embedded analytics for non-technical users

📫 Contact
Made by @inigosnz – feel free to reach out or contribute!
