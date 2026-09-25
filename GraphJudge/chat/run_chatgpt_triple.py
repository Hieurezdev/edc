import os
import asyncio
import json
import functools
import ast
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# Set API key and base URL
api_key = "empty"
api_base = "http://localhost:5000/v1"
model_name = "Qwen/Qwen3-4B-Instruct-2507"
openai_async_client = AsyncOpenAI(api_key=api_key, base_url=api_base)

# Read the text to be denoised
text = []
entity = []
# dataset = "rebel_sub"
# dataset = "GenWiki-Hard"
# dataset = "SCIERC"
dataset = "corpus10"
dataset_path = f'./datasets/Qwen3_result_{dataset}/'
Denoised_Iteration = 1
Graph_Iteration = 1

# Read denoised text
with open(dataset_path + f'Iteration{Denoised_Iteration}/test_denoised.target', 'r') as f:
    text = [l.strip() for l in f.readlines()]

# Read the corresponding entities
with open(dataset_path + f'Iteration{Denoised_Iteration}/test_entity.txt', 'r') as f:
    entity = [l.strip() for l in f.readlines()]

async def api_model(prompt, **kwargs):
    messages = [{"role": "user", "content": prompt}]
    response = await openai_async_client.chat.completions.create(
        model=model_name, messages=messages, temperature=0.1, **kwargs
    )
    return response.choices[0].message.content

async def _run_api(prompts, max_concurrent=8):
    semaphore = asyncio.Semaphore(max_concurrent)
    async def limited_api_model(prompt):
        async with semaphore:
            return await api_model(prompt)
    tasks = [limited_api_model(prompt) for prompt in prompts]
    answers = await tqdm.gather(*tasks)
    return answers

async def main():
    prompts = []
    for i in range(len(text)):
        prompt = (
                f"Mục tiêu:\nChuyển đổi văn bản thành một đồ thị ngữ nghĩa (dạng danh sách các bộ ba triples) với văn bản và các thực thể cho trước. "
                f"Nói cách cách khác, bạn cần tìm mối quan hệ giữa các thực thể dựa trên thông tin trong văn bản.\n"
                f"Lưu ý:\n1. Tạo ra càng nhiều bộ ba (triples) càng tốt. "
                f"2. Hãy chắc chắn rằng mỗi phần tử trong danh sách là một bộ ba (triple) có đúng 3 phần tử.\n\n"
                f"Dưới đây là hai ví dụ:\n"
                f"Ví dụ 1: \nText: \"Shotgate Thickets là một khu bảo tồn thiên nhiên ở Vương quốc Anh được điều hành bởi Essex Wildlife Trust.\"\n"
                f"Entity List: [\"Shotgate Thickets\", \"Khu bảo tồn thiên nhiên\", \"Vương quốc Anh\", \"Essex Wildlife Trust\"]\n"
                f"Semantic Graph: [[\"Shotgate Thickets\", \"loại hình bảo tồn\", \"Khu bảo tồn thiên nhiên\"], "
                f"[\"Shotgate Thickets\", \"quốc gia\", \"Vương quốc Anh\"], [\"Shotgate Thickets\", \"vận hành bởi\", \"Essex Wildlife Trust\"]]\n"
                f"Ví dụ 2:\nText: \"Tháp Eiffel tọa lạc tại Paris, Pháp, là một danh lam thắng cảnh nổi tiếng và là địa điểm thu hút khách du lịch. "
                f"Nó được thiết kế bởi kỹ sư Gustave Eiffel và hoàn thành vào năm 1889.\"\n"
                f"Entity List: [\"Tháp Eiffel\", \"Paris\", \"Pháp\", \"danh lam thắng cảnh\", \"Gustave Eiffel\", \"1889\"]\n"
                f"Semantic Graph: [[\"Tháp Eiffel\", \"tọa lạc tại\", \"Paris\"], [\"Tháp Eiffel\", \"tọa lạc tại\", \"Pháp\"], "
                f"[\"Tháp Eiffel\", \"là một\", \"danh lam thắng cảnh\"], [\"Tháp Eiffel\", \"được thiết kế bởi\", \"Gustave Eiffel\"], [\"Tháp Eiffel\", \"hoàn thành vào\", \"1889\"]]\n\n"
                f"Tham khảo các ví dụ trên và thực hiện yêu cầu sau:\nText: {text[i]}\nEntity List:{entity[i]}\nSemantic graph:"
            )
        prompts.append(prompt)

    responses = await _run_api(prompts)

    # 写入文件
    with open(dataset_path + f"Graph_Iteration{Graph_Iteration}/test_generated_graphs.txt", "w") as output_file:
        for response in responses:
            output_file.write(response.strip().replace('\n', '') + '\n')

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())