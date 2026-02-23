from openai import OpenAI


# Your tool implementation
def square_the_number(num: float) -> dict:
    return num ** 2


# Define Tools
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
                        "description": "input_num is a number that will be squared"
                    }
                },
            }
        }
    }
]


# Define LLM
client = OpenAI(
    # Use a custom endpoint compatible with OpenAI API
    base_url='http://llm.ai.r-ccs.riken.jp:11434/v1',  # api_base
    api_key="sk-ll-324kmjh8"
)

messages = [
    {"role": "system", "content": "You may call tools when helpful."},
    {'role': 'user', 'content': 'square the number 1024'}]

completion = client.chat.completions.create(
    messages=messages,
    model="zai-org/GLM-4.7-Flash", # Options: Kimi-K2-Thinking K2-Think zai-org/GLM-4.7-Flash qwen3-coder:30b
    max_tokens=16*1024,
    tools=tools,
    tool_choice="auto", # "required",  # allow the model to call the tool
)

print(completion.choices[0])
