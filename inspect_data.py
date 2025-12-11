import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# User mentioned: "hf://datasets/ScaleAI/PRBench/"
# and "finance_hard-00000-of-00001.parquet"

base_url = "hf://datasets/ScaleAI/PRBench/"
# Attempting to guess the filename for the finance split based on finance_hard
# Often these datasets have a pattern.
# 'finance_hard': 'data/finance_hard-00000-of-00001.parquet'
filenames = [
    "data/finance_hard-00000-of-00001.parquet",
    "data/finance-00000-of-00001.parquet" # Guessing this one exists
]

for fname in filenames:
    url = base_url + fname
    print(f"Trying to load {url}...")
    try:
        df = pd.read_parquet(url)
        print(f"Success loading {fname}")
        print("Columns:", df.columns.tolist())
        if 'topic' in df.columns:
            print("Unique Topics:", df['topic'].unique()[:10])
        
        row = df.iloc[0]
        print("\n--- Sample Row ---")
        if 'prompt_0' in df.columns:
            print("Prompt 0:", row['prompt_0'][:200])
        if 'response_0' in df.columns:
            print("Response 0:", str(row['response_0'])[:200])
        if 'reference_texts_0' in df.columns:
            print("Refs 0:", str(row['reference_texts_0'])[:200])
            
        print("-" * 20)
    except Exception as e:
        print(f"Failed to load {fname}: {e}")
