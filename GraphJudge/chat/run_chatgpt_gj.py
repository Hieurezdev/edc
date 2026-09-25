import os
import aiohttp
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm

# Set API key and base URL
api_key = "empty"
api_base = "http://localhost:5000/v1"
model_name = "Qwen/Qwen3-4B-Instruct-2507"

folder = "Qwen3_result_corpus10"
iteration = 1

# Input and output file paths
input_file = f"./datasets/{folder}/Graph_Iteration{iteration}/test_instructions_llama2_7b_itr{iteration}.csv"
output_file = f"./datasets/{folder}/Graph_Iteration{iteration}/pred_instructions_context_llama2_7b_itr{iteration}.csv"

# Load instructions from CSV file
df = pd.read_csv(input_file)
prompts_list = df["prompt"].tolist()

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def generate_prompt(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:"""

async def get_chatgpt_completion(session, instruction):
    """
    Send prompt to GPT/Local LLM and get the generated response.
    """
    url = f"{api_base}/chat/completions"
    prompt = generate_prompt(instruction)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    while True:
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error fetching completion: {e}")
            await asyncio.sleep(1)

async def process_instructions():
    """
    Process each instruction and generate responses using GPT/Local LLM.
    """
    print(f"Running batch graph judgment on '{input_file}'...")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for prompt in prompts_list:
            tasks.append(get_chatgpt_completion(session, prompt))

        # Execute all tasks and gather responses
        responses = await tqdm.gather(*tasks)

        # Clean responses (remove newlines like the original script)
        cleaned_responses = [resp.strip().replace('\n', ',') for resp in responses]

        # Save responses to a CSV file
        out_df = pd.DataFrame({
            "prompt": prompts_list,
            "generated": cleaned_responses
        })
        out_df.to_csv(output_file, index=False)
        print(f"Predictions written successfully to '{output_file}'")

if __name__ == "__main__":
    asyncio.run(process_instructions())