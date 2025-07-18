from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

import openai

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# response = client.responses.create(
#     model="gpt-4o",
#     input="Sort this list of numbers in Python and show the result: [5, 2, 9, 1, 5, 6]",
#     tools=[
#         {"type": "code_interpreter", "container": {"type": "auto"}}
#     ],
#     tool_choice="required",  # ⬅️ 關鍵！要求一定要用 tool
#     temperature=0.3,
# )

# # 分析結果
# for item in response.output:
#     if item.type == "code_interpreter_call":
#         print("💻 Code:\n", item.code)
#         if item.outputs:
#             for output in item.outputs:
#                 if output.type == "logs":
#                     print("📤 Output:\n", output.logs)
#     elif item.type == "message":
#         for c in item.content:
#             if c.type == "output_text":
#                 print("📝 Explanation:\n", c.text)
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hello world"}],
    temperature=0.3
)
print(response.choices[0].message.content)