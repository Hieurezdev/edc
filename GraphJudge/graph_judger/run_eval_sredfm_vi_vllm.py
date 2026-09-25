import os
import sys
import ast
import random
import asyncio
import argparse
import pandas as pd
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set seed for reproducibility
random.seed(42)

def generate_eval_instructions(source_file):
    """
    Generate positive and negative instruction data for graph judgment.
    """
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found at: {source_file}")

    triples = []
    with open(source_file, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            triples.append(ast.literal_eval(l.strip()))

    instructions_list = []
    for triple_list in triples:
        tail_list = [x[-1] for x in triple_list]
        for idx in range(len(triple_list)):
            if len(triple_list[idx]) < 2:
                continue
            elif len(triple_list[idx]) == 2:
                inst_pos = f"Is this true: {triple_list[idx][0]} {triple_list[idx][1]}"
                instructions_list.append({
                    "instruction": inst_pos,
                    "label": True, # True for positive
                    "raw_triple": triple_list[idx]
                })
            else:
                # positive instance
                inst_pos = f"Is this true: {triple_list[idx][0]} {triple_list[idx][1]} {triple_list[idx][2]}?"
                instructions_list.append({
                    "instruction": inst_pos,
                    "label": True,
                    "raw_triple": triple_list[idx]
                })
                
                # negative instance----randomly select tail entity
                neg_tail_list = [x for x in tail_list if x != triple_list[idx][2]]
                if len(neg_tail_list) >= 1:
                    neg_tail = random.choice(neg_tail_list)
                    inst_neg = f"Is this true: {triple_list[idx][0]} {triple_list[idx][1]} {neg_tail}?"
                    instructions_list.append({
                        "instruction": inst_neg,
                        "label": False, # False for negative
                        "raw_triple": [triple_list[idx][0], triple_list[idx][1], neg_tail]
                    })
    return instructions_list

def generate_prompt(instruction):
    return f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response:
"""

async def query_vllm(client, model_name, prompt, semaphore):
    async with semaphore:
        try:
            response = await client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=64,
                temperature=0,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            return f"Error: {e}"

async def evaluate_split(client, model_name, instructions, output_csv_path, concurrency_limit=64):
    prompts = [generate_prompt(item["instruction"]) for item in instructions]
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    tasks = [
        query_vllm(client, model_name, prompt, semaphore)
        for prompt in prompts
    ]
    
    generated_responses = await tqdm.gather(*tasks, desc="Querying vLLM Server")
    
    # Process predictions
    pred_labels = []
    for resp in generated_responses:
        resp_lower = resp.lower().strip()
        # Classify as False if response starts with or contains 'no', 'false', 'không' or 'sai'.
        if "no" in resp_lower[:15] or "false" in resp_lower[:15] or "không" in resp_lower[:15] or "sai" in resp_lower[:15]:
            pred_labels.append(False)
        else:
            pred_labels.append(True)
            
    gold_labels = [item["label"] for item in instructions]
    raw_instructions = [item["instruction"] for item in instructions]
    raw_triples = [str(item["raw_triple"]) for item in instructions]
    
    # Save predictions to CSV
    df = pd.DataFrame({
        "instruction": raw_instructions,
        "raw_triple": raw_triples,
        "gold_label": gold_labels,
        "pred_label": pred_labels,
        "generated_response": generated_responses
    })
    df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"Predictions saved to: {output_csv_path}")
    
    # Calculate metrics
    accuracy = accuracy_score(gold_labels, pred_labels)
    precision = precision_score(gold_labels, pred_labels)
    recall = recall_score(gold_labels, pred_labels)
    f1 = f1_score(gold_labels, pred_labels)
    
    return accuracy, precision, recall, f1

async def main():
    parser = argparse.ArgumentParser(description="Evaluate sredfm_vi on vLLM server.")
    parser.add_argument("--host", type=str, default="localhost", help="vLLM server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port (default: 8000)")
    parser.add_argument("--concurrency", type=int, default=64, help="Maximum concurrent requests (default: 64)")
    parser.add_argument("--dataset_dir", type=str, default="../datasets/sredfm_vi", help="Dataset directory path")
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}/v1"
    print(f"Connecting to vLLM server at: {base_url}...")
    client = AsyncOpenAI(base_url=base_url, api_key="none")
    
    try:
        models = await client.models.list()
        model_name = models.data[0].id
        print(f"Found active model on vLLM: {model_name}")
    except Exception as e:
        print(f"Failed to connect to vLLM server: {e}")
        sys.exit(1)
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(script_dir, args.dataset_dir))
    
    splits = ["validation", "test"]
    results = {}
    
    for split in splits:
        source_file = os.path.join(dataset_dir, f"{split}.source")
        output_csv = os.path.join(dataset_dir, f"pred_{split}_vllm.csv")
        
        print(f"\n==========================================")
        print(f"Processing split: {split}")
        print(f"==========================================")
        
        try:
            instructions = generate_eval_instructions(source_file)
            print(f"Generated {len(instructions)} evaluation instructions.")
            
            acc, prec, rec, f1 = await evaluate_split(client, model_name, instructions, output_csv, args.concurrency)
            results[split] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1
            }
            
            print(f"\nResults for {split}:")
            print(f"  Accuracy:  {acc:.4%}")
            print(f"  Precision: {prec:.4%}")
            print(f"  Recall:    {rec:.4%}")
            print(f"  F1-Score:  {f1:.4%}")
            
        except Exception as e:
            print(f"Error processing split {split}: {e}")
            
    print("\n========================= FINAL SUMMARY =========================")
    for split, metrics in results.items():
        print(f"Split: {split}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4%}")
    print("================================================================")

if __name__ == "__main__":
    asyncio.run(main())
