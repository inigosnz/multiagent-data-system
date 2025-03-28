# 🤖 Multi-Agent System for Data Interaction and Management

This project presents a comprehensive multi-agent system designed to enhance the way users interact with and manage complex datasets, particularly in industrial or IoT contexts. By combining state-of-the-art techniques from Big Data processing and Artificial Intelligence, the system aims to bridge the gap between human language and data manipulation — making advanced analytics accessible to users regardless of their technical background.

At the heart of the system lies a network of intelligent agents powered by Large Language Models (LLMs), which enable high-level natural language understanding and task automation. The primary objective is to allow users to write queries in plain English — such as “Show me all records with high torque and tool wear over 200 minutes” — and receive structured, accurate, and actionable results without writing a single line of code.

The architecture is fully modular and agent-driven. Each agent in the system is responsible for a specific step in the pipeline:

- **Query Extraction Agent** interprets the user’s input using domain-specific knowledge and dataset documentation, extracting structured filtering conditions.
- **JSON Formatter Agent** translates those conditions into a standardized JSON format that other agents can process.
- **Code Generation Agent** uses the structured query to produce executable Pandas code for filtering and retrieving relevant data.
- **Code Verifier Agent** checks the generated code for correctness and safety, using an LLM to explain any issues in plain language.
- **Code Execution Agent** securely runs the code and presents the results in a clean, responsive Streamlit interface.

This multi-agent collaboration enables a complete natural-language-to-data pipeline. It allows users to perform sophisticated data analysis tasks — such as identifying failure patterns, filtering by formulas, or extracting KPIs — with just a sentence.

Key design principles of the system include:
- **Transparency**: All code and transformations are visible to the user.
- **Scalability**: The architecture can support new agents (e.g., for SQL generation, document search, or charting) and adapt to new datasets.
- **Privacy and Portability**: The use of locally hosted LLMs (via Ollama) ensures full offline capability and data confidentiality.
- **User-Centered Design**: Built with Streamlit, the interface is minimalistic and intuitive, requiring zero technical expertise from end users.

This platform is not only a proof of concept for agent-based data systems but also a functional toolkit for any environment where structured data needs to be queried, explored, and understood interactively. It is ideal for industrial monitoring, exploratory analysis, quality control, or any scenario where fast, flexible, and explainable data interaction is critical.

---

## 📁 Project Structure
```bash
multiagent-data-system/
│
├── src/                             # Main application source
│   ├── agents/                      # Modular LLM agent logic
│   │   ├── code_executor.py
│   │   ├── code_verifier.py
│   │   ├── json_formatter.py
│   │   ├── pandas_coder.py
│   │   └── query_extractor.py
│   │
│   ├── data/                        # Dataset and documentation
│   │   ├── IOT.csv
│   │   ├── dataset_description.txt
│   │   └── Technical_documentation.txt
│   │
│   └── main.py                      # Streamlit UI entry point
│
├── .gitignore                       # Git ignore rules
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```
---

## 🚀 How to Run

### 1. Install the dependencies:

pip install -r requirements.txt

### 2. Launch the app:

streamlit run src/main.py

Make sure you have a local Ollama server running with the required models (e.g., deepseek-r1:7b, qwen2.5-coder:7b).

## 🧠 Key Features
Natural language query interpretation

Condition extraction and logical formatting (in JSON)

Automatic generation of valid Pandas code

LLM-powered error detection and explanations

Safe code execution and interactive result display

