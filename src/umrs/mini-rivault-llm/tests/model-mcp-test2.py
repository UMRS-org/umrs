import os

from openai import OpenAI


def square_the_number(input_num: float) -> dict:
    return {"result": input_num**2}


client = OpenAI(
    base_url="http://llm.ai.r-ccs.riken.jp:11434/v1",
    api_key=os.environ["RCCS_LLM_API_KEY"],
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "square_the_number",
            "description": "output the square of the number.",
            "parameters": {
                "type": "object",
                "required": ["input_num"],
                "properties": {
                    "input_num": {
                        "type": "number",
                        "description": "input number to square",
                    }
                },
            },
        },
    }
]

messages = [{"role": "user", "content": "square the number 1024"}]

completion = client.chat.completions.create(
    messages=messages,
    model="qwen3-coder:30b",
    tools=tools,
)

choice = completion.choices[0]
print(choice)

import re


def extract_input_num(text):
    match = re.search(r"<parameter=input_num>\s*(.*?)\s*</parameter>", text)
    return float(match.group(1)) if match else None


choice = completion.choices[0]
content = choice.message.content

if "<tool_call>" in content:
    input_num = extract_input_num(content)
    if input_num is not None:
        result = square_the_number(input_num)
        print("Tool result:", result)
