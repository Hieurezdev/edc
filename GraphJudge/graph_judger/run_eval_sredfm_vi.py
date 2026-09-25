import os
import sys
sys.modules['transformer_engine'] = None
import ast
import random
import torch
import pandas as pd
from tqdm import tqdm
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set seed for reproducibility
random.seed(42)

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = "NousResearch/Llama-2-7b-hf"
LORA_WEIGHTS = os.path.join(script_dir, "models", "llama2-7b-lora-rebel-sub")
DATASET_DIR = os.path.abspath(os.path.join(script_dir, "../datasets/sredfm_vi"))

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

def evaluate_split(model, tokenizer, instructions, output_csv_path, batch_size=32):
    prompts = [generate_prompt(item["instruction"]) for item in instructions]
    total_num = len(prompts)
    
    generated_responses = []
    
    for i in tqdm(range(0, total_num, batch_size), desc="Running Inference"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=GenerationConfig(
                    num_beams=5,
                    do_sample=False,
                ),
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        for s in generation_output:
            output = tokenizer.decode(s, skip_special_tokens=True)
            if "### Response:" in output:
                response = output.split("### Response:")[1].strip()
            else:
                response = output.strip()
            generated_responses.append(response)
            
    # Process predictions
    pred_labels = []
    for resp in generated_responses:
        resp_lower = resp.lower().strip()
        # If response starts with or contains 'no', 'false', 'không' or 'sai', classify as False.
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

def main():
    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading base model...")
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading PEFT/LoRA adapter weights...")
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    splits = ["validation", "test"]
    results = {}
    
    for split in splits:
        source_file = os.path.join(DATASET_DIR, f"{split}.source")
        output_csv = os.path.join(DATASET_DIR, f"pred_{split}_eval.csv")
        
        print(f"\n==========================================")
        print(f"Processing split: {split}")
        print(f"==========================================")
        
        try:
            instructions = generate_eval_instructions(source_file)
            print(f"Generated {len(instructions)} evaluation instructions.")
            
            acc, prec, rec, f1 = evaluate_split(model, tokenizer, instructions, output_csv)
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
    main()
