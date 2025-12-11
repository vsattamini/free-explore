# Financial RAG Assistant

A Retreival Augmented Generation (RAG) agent specialized in finance, powered by Gemini and ChromaDB.  
It uses the **ScaleAI/PRBench** dataset to answer complex financial questions, capable of running Python code for calculations.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- A Google Cloud API Key for Gemini.

### 2. Environment Setup
Create a `.env` file in the root directory and add your key:
```bash
GOOGLE_API_KEY=your_api_key_here
```

### 3. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 4. Data Ingestion (Build Knowledge Base)
Before running the agent, you need to download and index the financial data.
This script downloads `PRBench/finance_hard`, embeds the prompt + scratchpad, and stores it in a local ChromaDB.
```bash
python rag_engine.py
```
*Note: This might take a moment. By default, it indexes a subset of data for speed.*

### 5. Run the Application
Launch the Gradio interface:
```bash
python app.py
```
The application will be available at: **http://localhost:7860**

## 🏗 Architecture

- **Data Source**: [ScaleAI/PRBench](https://huggingface.co/datasets/ScaleAI/PRBench) (Finance Split).
- **Embeddings**: Gemini `text-embedding-004`.
- **Vector Store**: ChromaDB (Persistent).
- **LLM**: Gemini 2.5 Flash.
- **Interface**: Gradio.

### Features
- **Topic Filtering**: Restrict retrieval to specific financial domains (e.g., "Corporate Finance", "Risk Management").
- **Hybrid Agent**: Can retrieve knowledge AND execute Python code for calculations.
- **RAG Strategy**: Embeds the expert "Prompt + Scratchpad" to align with user queries, and retrieves the expert "Response".
