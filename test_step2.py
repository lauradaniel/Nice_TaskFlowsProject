#!/usr/bin/env python3
import sys
import pandas as pd
from intents import IntentBuilder, categories2dataframe
import bedrock
from default_prompts import STEP2_ASSIGN_CATEGORIES_PROMPT

print("="*60)
print("TESTING STEP 2 - CATEGORIZATION")
print("="*60)

# Use the files from your last run
WORK_DIR = "./data/intent_generation_1760969836"  # CHANGE THIS to your actual directory
STEP1_CSV = f"{WORK_DIR}/step1_intents.csv"
CATEGORIES_TXT = "./uploads/mapping_1760914066_categories.txt"

print("\n📋 Loading Step 1 results...")
reasons_df = pd.read_csv(STEP1_CSV, sep='\t').dropna(subset=['Intent']).copy()
print(f"   Loaded {len(reasons_df)} intents")
print(f"   Sample intents:")
for i in range(min(5, len(reasons_df))):
    print(f"     {i}: {reasons_df.iloc[i]['Intent']}")

reasons_df['Ind'] = range(len(reasons_df))

print("\n🏷️  Categorizing FIRST 10 intents...")
client = bedrock.get_client(region="us-east-1")
builder = IntentBuilder(client, cluster_prompt="", assign_prompt=STEP2_ASSIGN_CATEGORIES_PROMPT)

# Test with just the first 10
test_intents = list(reasons_df.iloc[0:10]['Intent'])
print(f"   Test intents: {test_intents[:3]}...")

try:
    result = builder.assign_reasons(
        CATEGORIES_TXT,
        test_intents,
        'anthropic.claude-3-5-sonnet-20240620-v1:0',
        0
    )
    
    print("\n📊 LLM Response:")
    print(result)
    print("\n" + "="*60)
    
    # Try to parse it
    from io import StringIO
    data_io = StringIO(result)
    assign_cols = ['Ind', 'Intent_Input', 'Intent_Category', 'L3_Score', 'L2_Score', 'L1_Score']
    df = pd.read_csv(data_io, names=assign_cols, on_bad_lines='skip')
    
    print(f"\n✅ Parsed {len(df)} rows")
    print(df.head())
    
except Exception as e:
    print(f"\n❌ Failed: {e}")
    import traceback
    traceback.print_exc()