#!/usr/bin/env python
# coding: utf-8

import os
import sys
import csv
import json
import ast
import random
import argparse
import pandas as pd
from tqdm import tqdm

def extract_list_from_text(text):
    # Try literal split first
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part_clean = part.strip()
            # If it's a code block with language spec like "```json" or "```python"
            if part_clean.startswith("json"):
                part_clean = part_clean[4:].strip()
            elif part_clean.startswith("python"):
                part_clean = part_clean[6:].strip()
            if part_clean.startswith("[") and part_clean.endswith("]"):
                try:
                    return ast.literal_eval(part_clean)
                except Exception:
                    pass
    # If that fails or isn't present, find the first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        list_str = text[start:end+1]
        try:
            return ast.literal_eval(list_str)
        except Exception:
            pass
    # Fallback/safety
    return [['none', 'none', 'none']]

def generate_instructions(dataset_path, split="train"):
    """
    Generate positive and negative instruction data for graph judgment.
    Reads from: {dataset_path}/{split}.source
    Writes to: {dataset_path}/{split}_instructions_llama.json
    """
    print(f"Generating instructions for split '{split}' in dataset '{dataset_path}'...")
    source_file = os.path.join(dataset_path, f"{split}.source")
    output_file = os.path.join(dataset_path, f"{split}_instructions_llama.json")

    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found at: {source_file}")

    triples = []
    with open(source_file, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            triples.append(ast.literal_eval(l.strip()))

    res_list = []
    for triple_list in tqdm(triples, desc=f"Processing {split} source"):            
        tail_list = [x[-1] for x in triple_list]
        for idx in range(len(triple_list)):
            if len(triple_list[idx]) == 1:
                continue
            elif len(triple_list[idx]) == 2:
                inst_pos = f"Is this true: {triple_list[idx][0]} {triple_list[idx][1]}"
                output_pos = "Yes, this is true."
                temp_dict_pos = {"instruction": inst_pos, "input": "", "output": output_pos}
                res_list.append(temp_dict_pos)
            else:
                # positive instance
                inst_pos = f"Is this true: {triple_list[idx][0]} {triple_list[idx][1]} {triple_list[idx][2]}?"
                output_pos = "Yes, this is true."
                temp_dict_pos = {"instruction": inst_pos, "input": "", "output": output_pos}
                res_list.append(temp_dict_pos)
                
                # negative instance----randomly select tail entity
                neg_tail_list = [x for x in tail_list if x != triple_list[idx][2]]
                if len(neg_tail_list) >= 1:
                    neg_tail = random.choice(neg_tail_list)
                    inst_neg = f"Is this true: {triple_list[idx][0]} {triple_list[idx][1]} {neg_tail}?"
                    output_neg = "No, this is not true."
                    temp_dict_neg = {"instruction": inst_neg, "input": "", "output": output_neg}
                    res_list.append(temp_dict_neg)
                
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(res_list, f, indent=4)
    print(f"Instructions successfully written to: {output_file}")


def format_generated_graphs(dataset_path, iteration, is_gpt_baseline=False):
    """
    Format generated graph txt to CSV text format for evaluation.
    """
    if is_gpt_baseline:
        input_file = os.path.join(dataset_path, "gpt_baseline", "test_generated_graphs.txt")
        output_file = os.path.join(dataset_path, "gpt_baseline", "test_instructions_llama2_7b_gpt.csv")
    else:
        input_file = os.path.join(dataset_path, f"Graph_Iteration{iteration}", "test_generated_graphs.txt")
        output_file = os.path.join(dataset_path, f"Graph_Iteration{iteration}", f"test_instructions_llama2_7b_itr{iteration}.csv")

    print(f"Formatting generated graphs from '{input_file}' to '{output_file}'...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Generated graphs file not found at: {input_file}")

    triples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            triples += extract_list_from_text(l.strip())

    # Ensure parent directory of output exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['prompt', 'response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for triple in tqdm(triples, desc="Formatting triples"):
            if is_gpt_baseline:
                triple = [str(x) for x in triple]
            if len(triple) != 3:
                prompt = f"Is this true: {' '.join(triple)}?"
            else:
                subject, predicate, obj = triple
                prompt = f"Is this true: {subject} {predicate} {obj}?"
            response = "**"
            writer.writerow({'prompt': prompt, 'response': response})

    print(f"CSV file created successfully at: {output_file}")


def filter_generated_graphs(dataset_path, iteration, is_gpt_baseline=False, use_simplebase=False, pred_model="llama3_8b"):
    """
    Remove incorrect triples from generated graphs based on evaluation predictions.
    """
    if is_gpt_baseline:
        input_file = os.path.join(dataset_path, "gpt_baseline", "test_generated_graphs.txt")
        pred_file = os.path.join(dataset_path, "gpt_baseline", "pred_instructions_context_llama2_7b_woECTD.csv")
        output_file = os.path.join(dataset_path, "gpt_baseline", "test_generated_graphs_final.txt")
        limit = 10
    else:
        input_file = os.path.join(dataset_path, f"Graph_Iteration{iteration}", "test_generated_graphs.txt")
        limit = 100
        if use_simplebase:
            pred_file = os.path.join(dataset_path, f"Graph_Iteration{iteration}", f"pred_instructions_context_llama2_7b_itr{iteration}_simplebase.csv")
            output_file = os.path.join(dataset_path, f"Graph_Iteration{iteration}", "test_generated_graphs_final_simplebase.txt")
        else:
            pred_file = os.path.join(dataset_path, f"Graph_Iteration{iteration}", f"pred_instructions_context_{pred_model}_itr{iteration}.csv")
            output_file = os.path.join(dataset_path, f"Graph_Iteration{iteration}", f"test_generated_graphs_{pred_model}_final.txt")

    print(f"Filtering '{input_file}' using predictions '{pred_file}' to create '{output_file}'...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Generated graphs file not found at: {input_file}")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found at: {pred_file}")

    triples = []
    triples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            triples.append(extract_list_from_text(l.strip()))

    pred_res = pd.read_csv(pred_file, header=0, sep=',')
    res_list = []
    for index, data in tqdm(pred_res.iterrows(), desc="Analyzing predictions"):
        try:
            response = data['generated'].lower()
            if 'no' in response[:limit] or 'false' in response[:limit]:
                res_list.append(False)
            else:
                res_list.append(True)
        except:
            res_list.append(False)

    new_triples = []
    i = 0
    for triple_list in triples:
        new_triple_list = []
        for triple in triple_list:
            if i < len(res_list):
                if res_list[i]:
                    new_triple_list.append(triple)
            i += 1
        if is_gpt_baseline and len(new_triple_list) < 1:
            new_triple_list = triple_list
        new_triples.append(new_triple_list)

    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in new_triples:
            f.write('```' + str(doc).replace('\n', '') + '```' + '\n')
            
    print(f"Filtered graphs written successfully to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare KG completion instructions, format graphs, or filter graphs.")
    parser.add_argument(
        "--task",
        choices=["gen-instructions", "format-graphs", "filter-graphs"],
        required=True,
        help="The preparation task to execute."
    )
    parser.add_argument(
        "--dataset_path",
        default="./GPT4o_mini_result_rebel_sub/",
        help="Path to the dataset directory (e.g. ./GPT4o_mini_result_rebel_sub/)."
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train",
        help="Data split to use when generating instructions (default: train)."
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Graph iteration index (default: 1)."
    )
    parser.add_argument(
        "--is_gpt_baseline",
        action="store_true",
        help="Flag indicating if the target is GPT baseline."
    )
    parser.add_argument(
        "--use_simplebase",
        action="store_true",
        help="Use simple baseline settings during graph filtering."
    )
    parser.add_argument(
        "--pred_model",
        default="llama3_8b",
        help="Model name identifier for predicting judgment labels (default: llama3_8b)."
    )

    args = parser.parse_args()

    if args.task == "gen-instructions":
        generate_instructions(args.dataset_path, split=args.split)
    elif args.task == "format-graphs":
        format_generated_graphs(args.dataset_path, args.iteration, args.is_gpt_baseline)
    elif args.task == "filter-graphs":
        filter_generated_graphs(
            args.dataset_path,
            args.iteration,
            is_gpt_baseline=args.is_gpt_baseline,
            use_simplebase=args.use_simplebase,
            pred_model=args.pred_model
        )

if __name__ == "__main__":
    main()
