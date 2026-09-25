import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

api_key = "empty"
api_base = "http://localhost:8000/v1"
model_name = "Qwen/Qwen3-4B-Instruct-2507"

# Read the text
folder = "GPT4o_mini_result_corpus10"
with open(f'./datasets/{folder}/test.target', 'r') as f:
    text = [l.strip() for l in f.readlines()]

async def api_model(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    response = await openai_async_client.chat.completions.create(
        model=model_name, messages=messages, temperature=0, **kwargs
    )

    return response.choices[0].message.content

async def _run_api(queries, max_concurrent=16):
    semaphore = asyncio.Semaphore(max_concurrent)  # Limit maximum concurrency to 8

    async def limited_api_model(query):
        async with semaphore:
            return await api_model(query)

    tasks = [limited_api_model(query) for query in queries]
    answers = await tqdm.gather(*tasks)
    return answers

async def process_texts():
    queries = []
    for t in text:
        prompt = (
            f"""
Mục tiêu:
Chuyển đổi văn bản thành một đồ thị ngữ nghĩa (dạng danh sách các bộ ba triples).

Ví dụ 1:
Văn bản: "Shotgate Thickets là một khu bảo tồn thiên nhiên ở Vương quốc Anh được điều hành bởi Essex Wildlife Trust."
Semantic Graph: 
```[["Shotgate Thickets", "loại hình bảo tồn", "Khu bảo tồn thiên nhiên"], ["Shotgate Thickets", "quốc gia", "Vương quốc Anh"], ["Shotgate Thickets", "vận hành bởi", "Essex Wildlife Trust"]]```
Ví dụ 2:
Văn bản: "Tháp Eiffel tọa lạc tại Paris, Pháp, là một danh lam thắng cảnh nổi tiếng và là địa điểm thu hút khách du lịch. Nó được thiết kế bởi kỹ sư Gustave Eiffel và hoàn thành vào năm 1889."
Semantic Graph:
```[["Tháp Eiffel", "tọa lạc tại", "Paris"], ["Tháp Eiffel", "tọa lạc tại", "Pháp"], ["Tháp Eiffel", "là một", "danh lam thắng cảnh"], ["Tháp Eiffel", "được thiết kế bởi", "Gustave Eiffel"], ["Tháp Eiffel", "hoàn thành vào", "1889"]]```

Lưu ý: 
1. Hãy chắc chắn rằng mỗi phần tử trong danh sách là một bộ ba (triple) có đúng 3 phần tử.
2. Trả về kết quả dưới dạng một đồ thị ngữ nghĩa (semantic graph) giống hệt ví dụ trên (danh sách các danh sách con).
3. KHÔNG trả về định dạng kiểu '```json[ .. ]```', định dạng bắt buộc phải là dạng '```[...]```'.

Văn bản: "{t}"
Semantic Graph:
"""
        )
        queries.append(prompt)

    responses = await _run_api(queries)

    with open(f"./datasets/{folder}/gpt_baseline/test_generated_graphs.txt", "w") as output_file:
        for response in responses:
            output_file.write(response.strip().replace('\n', '') + '\n')

# Run the async process
asyncio.run(process_texts())