"""
Default prompts for intent generation and assignment
"""

STEP1_GENERATE_INTENTS_PROMPT = """You are analyzing customer service conversations for {company_name}.

Company Description: {company_description}

Analyze the conversation and identify the primary customer intent.

Conversation:
{conv}

Respond in this EXACT format:
Intent description,1

Where "Intent description" is the customer's primary goal in 5-10 words.

Examples:
Inquiring about account balance,1
Requesting password reset,1
Updating billing address information,1
Reporting technical issue with system,1

Your response (must include comma and number):"""

# Prompt for Step 2: Assign Categories
STEP2_ASSIGN_CATEGORIES_PROMPT = """You are categorizing customer service intents into a predefined taxonomy.

Given the following intent categories:
{categories}

Please categorize each of the following intents. For each intent, provide:
1. The line number (index)
2. The original intent text
3. The best matching L3 category from the taxonomy above (format: L1 - L2 - L3)
4. L3 confidence score (1-5, where 5 is perfect match)
5. L2 confidence score (1-5)
6. L1 confidence score (1-5)

Intents to categorize:
{reasons_str}

Respond in CSV format with NO HEADER:
index,intent,category,l3_score,l2_score,l1_score

Example output:
0,Update billing address,ACCOUNT - Account Management - Update Account Information,5,5,5
1,Report fraud,SECURITY - Fraud & Security - Report Fraudulent Activity,5,5,5

Your response (CSV only, no additional text):"""

