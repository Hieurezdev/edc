import openai
import json
import re
from project_variables import OPENAI_KEY, OPENAI_MODEL, MAX_TOKENS, TEMPERATURE

# Replace with your actual OpenAI API key
openai.api_key = OPENAI_KEY

# Define the prompt
prompt = """
- Hãy chuẩn hóa từ {nsw} thành dạng chuẩn

- Giải nghĩa từ {nsw} với từng dạng chuẩn

- Thêm một số biến thể viết tắt cho mỗi dạng chuẩn của {nsw}  (nếu có)

- Đặt một câu ví dụ với từ không chuẩn {nsw}.

Kết quả trả về ở dạng HTML.
"""

def run_chatgpt(nsw):

    filled_prompt = prompt.format(nsw=nsw)

    # Generate text using the ChatGPT 'completions' API with the new syntax
    response = openai.chat.completions.create(
            model=OPENAI_MODEL,  # or use "gpt-4" if you have access to it
            messages=[{"role": "user", "content": filled_prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )

    # Get the generated text from the response
    response_dict = response.model_dump()    # <--- convert to dictionary
    response_message = response_dict["choices"][0]["message"]["content"]
    result = response_message

    return result



