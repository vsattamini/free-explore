import gradio as gr
from agent import FinancialAgent
import time

# Initialize agent
# Note: If the ingestion is still running in the background, this might
# wait or hit a lock depending on ChromaDB's implementation for the specific version.
print("Initializing Agent...")
agent = FinancialAgent()
print("Agent initialized.")

def chat_interface(message, history, topic):
    if not message:
        return ""
    
    # Simple spinner simulation or just wait
    response = agent.answer(message, topic_filter=topic)
    return response

# Get topics for dropdown
try:
    topics = agent.get_topics()
    topics = ["All"] + topics
except Exception as e:
    print(f"Error fetching topics (DB might be locked or empty): {e}")
    topics = ["All", "General"]

with gr.Blocks(title="Financial RAG Assistant") as demo:
    gr.Markdown(
        """
        # 🏦 Financial RAG Assistant
        Ask questions about complex financial topics. The agent uses `PRBench` data.
        """
    )
    
    with gr.Row():
        topic_dropdown = gr.Dropdown(
            choices=topics, 
            value="All", 
            label="Filter by Topic", 
            info="Restrict the knowledge retrieval to a specific financial domain."
        )
    
    chatbot = gr.ChatInterface(
        fn=chat_interface,
        additional_inputs=[topic_dropdown],
        examples=[
            ["What is tail dependence?", "All"],
            ["How do I calculate Free Cash Flow?", "Corporate Finance"],
            ["Explain stress testing for credit risk.", "Risk Management & Stress Testing"]
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, theme=gr.themes.Soft())
