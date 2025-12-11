from google import genai
from google.genai import types
import os
import io
import sys
import traceback
from rag_engine import FinancialRAG

# Ensure API key is set
# The user said the key is in .env GOOGLE_API_KEY
# google-genai Client automatically looks for GEMINI_API_KEY or GOOGLE_API_KEY?
# User snippet says GEMINI_API_KEY. But we have GOOGLE_API_KEY.
# We will pass it explicitly.

class FinancialAgent:
    def __init__(self):
        self.rag = FinancialRAG()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
             raise ValueError("GOOGLE_API_KEY not found.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash" # Trying 2.0-flash-exp as 2.5-flash might be typo or strict beta.
     
        
        # self.model_name = "gemini-2.5-flash" 

        # Define tools
        # In google-genai, tools are passed to config.
        # We define python functions and pass them.
        
        # Tool: Retrieve
        # We need to wrap the RAG query in a function
        def retrieve_financial_info(query: str, topic: str = "All") -> str:
            """Retrieve financial concepts, definitions, and insights from the knowledge base."""
            return self._rag_tool(query, topic)
            
        def execute_python_code(code: str) -> str:
            """Execute Python code to perform calculations or data analysis."""
            return self._execute_python(code)
            
        self.tools = [retrieve_financial_info, execute_python_code]
        
    def _rag_tool(self, query, topic):
        results = self.rag.query(query, topic_filter=topic)
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        context = ""
        for i, doc in enumerate(docs):
            context += f"\nSnippet {i+1} (Topic: {metas[i]['topic']}):\n{doc}\n"
        return context

    def _execute_python(self, code):
        buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer
        import pandas as pd # Ensure available in exec scope
        try:
            local_scope = {}
            # Allow some imports
            allowed_globals = {'__builtins__': __builtins__, 'pd': pd, 'import': __import__}
            exec(code, allowed_globals, local_scope)
            output = buffer.getvalue()
            return output if output else "Code executed successfully (no output)."
        except Exception as e:
            return f"Error executing code: {traceback.format_exc()}"
        finally:
            sys.stdout = original_stdout

    def get_topics(self):
        return self.rag.get_topics()

    def answer(self, user_message, topic_filter="All"):
        # Create a chat session (or just generate content with history if needed)
        # google-genai has client.chats.create()
        
        # To simplify, we'll just do a single turn or stateless approach for now, 
        # OR adapt to chat.
        # The App assumes a chat interface.
        
        # We can use the simple generate_content, but chat is better.
        # chat = self.client.chats.create(model=self.model_name)
        # But we need tools config.
        
        conf = types.GenerateContentConfig(
            tools=self.tools,
            temperature=0.1, # Low temp for financial
            system_instruction="You are a helpful Financial Assistant. Use the retrieve_financial_info tool to find information from the knowledge base. Use execute_python_code for calculations."
        )
        
        # We will use a fresh chat session for each query in this stateless wrapper 
        # (since Gradio ChatInterface passes history, but we aren't maintaining state object easily here without refactoring App to pass state).
        # Actually Gradio ChatInterface re-sends history? 
        # For this turn-based function, let's just send the message. 
        # If we want history, we'd need to convert Gradio history to Gemini history.
        # User requested "simple RAG".
        
        # Add topic context to message
        msg_with_context = f"{user_message} (Context Filter: {topic_filter})"
        
        # Direct generation with tools (automatic function calling!)
        # automatic_function_calling is default enabled in some versions, or we set it.
        
        # Update config to enable automatic tool execution
        conf.tools = self.tools
        conf.automatic_function_calling = types.AutomaticFunctionCallingConfig(disable=False)


        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=msg_with_context,
                config=conf
            )
            return response.text
        except Exception as e:
            # Fallback for model name errors
            if "404" in str(e):
                return f"Error: Model {self.model_name} not found. Please check API key access or model name."
            return f"Error generation: {e}"
