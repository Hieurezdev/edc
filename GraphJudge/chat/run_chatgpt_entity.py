import os
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

api_key = "empty"
api_base = "http://localhost:5000/v1"
model_name = "Qwen/Qwen3-4B-Instruct-2507"

# dataset = "rebel_sub"
# dataset = "GenWiki-Hard"  # rebel / webnlg / kelm
# dataset = "SCIERC"
dataset = "corpus10"
dataset_path = f'./datasets/Qwen3_result_{dataset}/'
Iteration = 1

if Iteration == 1:
    with open(dataset_path + 'test.target', 'r') as f:
        text = [l.strip() for l in f.readlines()]
else:
    with open(dataset_path + f'Iteration{Iteration - 1}/test_denoised.target', 'r') as f:
        text = [l.strip() for l in f.readlines()]

async def api_model(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    openai_async_client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    response = await openai_async_client.chat.completions.create(
        model=model_name, messages=messages, temperature=0, **kwargs
    )
    return response.choices[0].message.content

async def _run_api(queries, max_concurrent=8):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_api_model(query):
        async with semaphore:
            return await api_model(query)

    tasks = [limited_api_model(query) for query in queries]
    answers = await tqdm.gather(*tasks)
    return answers

async def extract_entities(texts):
    prompts = []
    for t in texts:
        prompt = f"""
Mục tiêu:
Chuyển đổi văn bản thành một danh sách các thực thể (entities).

Dưới đây là hai ví dụ:
Ví dụ 1:
Văn bản: "Shotgate Thickets là một khu bảo tồn thiên nhiên ở Vương quốc Anh được điều hành bởi Essex Wildlife Trust."
Danh sách thực thể: ["Shotgate Thickets", "Khu bảo tồn thiên nhiên", "Vương quốc Anh", "Essex Wildlife Trust"]
Ví dụ 2:
Văn bản: "Cung tròn có độ dài bằng bán kính gọi là cung có số đo 1 radian, gọi tắt là cung 1 radian. Góc ở tâm chắn cung 1 radian gọi là góc có số đo 1 radian, gọi tắt là góc 1 radian."
Danh sách thực thể: ["Cung tròn", "bán kính", "cung 1 radian", "Góc ở tâm", "góc 1 radian"]

Tham khảo các ví dụ trên và thực hiện yêu cầu sau:
Văn bản: "{t}"
Danh sách thực thể: """
        prompts.append(prompt)
    
    entities_list = await _run_api(prompts)
    return entities_list

async def denoise_text(texts, entities_list):
    prompts = []
    for t, entities in zip(texts, entities_list):
        prompt = f"""
Mục tiêu:
Loại bỏ các thông tin không liên quan khỏi văn bản thô dựa trên danh sách thực thể cho trước và viết lại văn bản đó một cách ngắn gọn, chuẩn hóa.

Dưới đây là hai ví dụ:
Ví dụ 1:
Văn bản thô: "Zakria Rezai (sinh ngày 29 tháng 7 năm 1989) là một cầu thủ bóng đá chơi cho câu lạc bộ Ordu Kabul F.C., một câu lạc bộ bóng đá của Afghanistan. Anh ấy cũng là tuyển thủ quốc gia Afghanistan và đã có 9 lần khoác áo đội tuyển. Anh ấy mặc áo số 14 và chơi ở vị trí trung vệ."
Thực thể: ["Zakria Rezai", "cầu thủ bóng đá", "Ordu Kabul F.C.", "Afghanistan", "29 tháng 7 năm 1989"]
Văn bản sau khi lọc: "Zakria Rezai là một cầu thủ bóng đá. Zakria Rezai chơi cho câu lạc bộ bóng đá Ordu Kabul F.C. Zakria Rezai có quốc tịch Afghanistan. Zakria Rezai sinh ngày 29 tháng 7 năm 1989. Ordu Kabul F.C. là một câu lạc bộ bóng đá có trụ sở tại Afghanistan."
Ví dụ 2:
Văn bản thô: "Cung tròn có độ dài bằng bán kính gọi là cung có số đo 1 radian, gọi tắt là cung 1 radian. Góc ở tâm chắn cung 1 radian gọi là góc có số đo 1 radian, gọi tắt là góc 1 radian."
Thực thể: ["Cung tròn", "bán kính", "cung 1 radian", "Góc ở tâm", "góc 1 radian"]
Văn bản sau khi lọc: "Cung tròn có độ dài bằng bán kính. Cung tròn có độ dài bằng bán kính có số đo 1 radian. Cung tròn có số đo 1 radian được gọi tắt là cung 1 radian. Góc ở tâm chắn cung 1 radian được gọi là góc 1 radian."

Tham khảo các ví dụ trên và thực hiện yêu cầu sau:
Văn bản thô: {t}
Thực thể: {entities}
Văn bản sau khi lọc: """
        prompts.append(prompt)
    
    denoised_texts = await _run_api(prompts)
    return denoised_texts

async def main():
    # 提取实体并保存
    entities_list = await extract_entities(text)
    with open(dataset_path + f"Iteration{Iteration}/test_entity.txt", "w") as output_file:
        for entities in entities_list:
            output_file.write(entities.strip().replace('\n', '') + '\n')
    
    # 读取提取的实体
    last_extracted_entities = []
    with open(dataset_path + f'Iteration{Iteration}/test_entity.txt', 'r') as f:
        for l in f.readlines():
            last_extracted_entities.append(l.strip())
    
    # 去噪文本并保存
    denoised_texts = await denoise_text(text, last_extracted_entities)
    with open(dataset_path + f"Iteration{Iteration}/test_denoised.target", "w") as output_file:
        for denoised_text in denoised_texts:
            output_file.write(denoised_text.strip().replace('\n', '') + '\n')

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())