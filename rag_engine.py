import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class FinancialRAG:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment or passed explicitly.")
        
        genai.configure(api_key=self.api_key)
        
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        # Use simple cosine similarity or whatever default
        self.collection_name = "finance_knowledge"
        
        # We will use Gemini for embeddings.
        # Chroma's Google wrapper might be handy, but doing it manually ensures control.
        # But let's try to use the built-in if available, or just a custom function class.
        self.embedding_fn = self._get_embedding_function()
        
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name, 
                embedding_function=self.embedding_fn
            )
        except Exception as e:
            # Fallback if collection doesn't exist
            print(f"Collection not found ({e}), creating...")
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )
            self._load_and_index_data()

    def _get_embedding_function(self):
        # Custom wrapper for Google Gemini Embeddings to work with Chroma
        class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self, api_key):
                self.api_key = api_key

            def __call__(self, input):
                # input is a list of strings
                # Gemini text-embedding-004 is good
                unique_inputs = list(set(input)) # Avoid duplicate calls if any
                # Process in batches if necessary, but for now simple
                # Gemini batch size is limited.
                
                # Check 
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=input,
                    task_type="retrieval_document",
                    title="Financial Document"
                )
                return result['embedding']
        
        return GeminiEmbeddingFunction(self.api_key)

    def _load_and_index_data(self):
        print("Loading data...")
        # Load datasets
        base_url = "hf://datasets/ScaleAI/PRBench/"
        files = [
            "data/finance_hard-00000-of-00001.parquet",
            "data/finance-00000-of-00001.parquet"
        ]
        
        all_dfs = []
        for f in files:
            try:
                df = pd.read_parquet(base_url + f)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not all_dfs:
            print("No data loaded.")
            return

        full_df = pd.concat(all_dfs, ignore_index=True)
        
        print(f"Indexing {len(full_df)} rows...")
        
        documents = []
        metadatas = []
        ids = []
        
        # We limit to first 500 rows to save time/quota if dataset is huge
        # User said "short on time".
        # But let's try to do reasonable amount.
        
        for idx, row in full_df.iterrows():
            if idx >= 100: # Limit for demo purposes as requested
                break
                
            content = row.get('response_0', '')
            # If response is empty, maybe try reference_texts_0
            if not content or str(content).strip() == "":
                content = str(row.get('reference_texts_0', ''))
            
            if not content or str(content).strip() == "[]":
                continue
                
            topic = row.get('topic', 'General')
            prompt = str(row.get('prompt_0', ''))
            scratchpad = str(row.get('scratchpad', ''))
            response = str(row.get('response_0', ''))
            
            # User request: Embed Prompt + Scratchpad
            # We will store the Response in metadata to retrieve it later without it influencing the embedding of the query matching.
            
            embed_text = f"Prompt: {prompt}\nScratchpad: {scratchpad}"
            
            documents.append(embed_text)
            
            # Store full content in metadata for the agent to read
            # Truncating response slightly if huge, but usually needed full.
            metadatas.append({
                "topic": topic, 
                "response": response, # Store response here
                "full_prompt": prompt[:500] # Just a snippet for meta info if needed
            })
            ids.append(f"doc_{idx}")
            
        if documents:
            # Add to collection
            batch_size = 20
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                try:
                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_meta,
                        ids=batch_ids
                    )
                except Exception as e:
                    print(f"Error batch indexing: {e}")
        
        print("Indexing complete.")

    def get_topics(self):
        # Retrieve unique topics from metadata
        # Chroma doesn't have a direct "get unique metadata values"
        # We can just return the predefined ones we saw or query all
        # For efficiency, I'll return a hardcoded list based on inspection + 'All'
        # Or cache it during ingest.
        # Let's peek into the collection or defaults.
        # Based on inspection:
        topics = [
            'Risk Management & Stress Testing',
            'International Finance & FX Hedging',
            'Market Microstructure, Trading & Liquidity',
            'Accounting & Financial Statement Analysis',
            'Taxation & Cross-Border Structuring',
            'Derivatives & Structured Products',
            'Wealth Management, Financial Planning & Advice',
            'FinTech, Crypto & Digital Assets',
            'Regulation, Compliance & Ethics',
            'Corporate Finance',
            'Investment Strategy & Portfolio Design',
            'Alternative Investments & Private Markets'
        ]
        return sorted(list(set(topics)))

    def query(self, query_text, topic_filter=None, n_results=3):
        where_filter = None
        if topic_filter and topic_filter != "All":
            where_filter = {"topic": topic_filter}
        
        # We need to embed the query first? 
        # Chroma collection.query handles embedding if we passed embedding_function to get_collection
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        return results

if __name__ == "__main__":
    rag = FinancialRAG()
    res = rag.query("What is tail dependence?", topic_filter="All")
    print(res)
