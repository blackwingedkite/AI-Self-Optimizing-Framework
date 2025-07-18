import google.generativeai as genai
from google.genai import types
from google.genai import Client as google_client
from dotenv import load_dotenv
import os
load_dotenv()
client = google_client(api_key=os.getenv("GOOGLE_API_KEY"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=["Please write a code to sort a list of numbers in ascending order. numbers = [5, 2, 9, 1, 5, 6]"],
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        temperature=0.4,
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
    )
)
print(response)