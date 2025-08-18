from openai import OpenAI
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
load_dotenv()

EVAL_OPENAI_BASE_URL = os.getenv("EVAL_OPENAI_BASE_URL")
EVAL_OPENAI_API_KEY = os.getenv("EVAL_OPENAI_API_KEY")
MAX_WORKERS = 10

PROMPT = """
You are a helpful assistant that can answer legal questions.

Here is the question:
{question}
"""

client = OpenAI(
    base_url = EVAL_OPENAI_BASE_URL,
    api_key=EVAL_OPENAI_API_KEY,
)


response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": PROMPT.format(question=question)}]
)

print(response.choices[0].message.content)