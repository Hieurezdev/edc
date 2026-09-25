#!/usr/bin/env python3
import os
import json
import pyarrow as pa
from tqdm import tqdm

def convert_split(arrow_path, out_dir, split_name):
    if not os.path.exists(arrow_path):
        print(f"File not found: {arrow_path}")
        return

    print(f"Processing {split_name} split from {arrow_path}...")
    with open(arrow_path, 'rb') as f:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
        df = table.to_pandas()

    os.makedirs(out_dir, exist_ok=True)
    
    # Map 'validation' to 'val' for target filenames, but we can write both for compatibility
    target_names = [split_name]
    if split_name == "validation":
        target_names = ["val", "validation"]

    for name in target_names:
        source_path = os.path.join(out_dir, f"{name}.source")
        target_path = os.path.join(out_dir, f"{name}.target")
        
        print(f"  Writing to {source_path} and {target_path}...")
        with open(source_path, 'w', encoding='utf-8') as f_source, open(target_path, 'w', encoding='utf-8') as f_target:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {name}"):
                text = row['text'].strip().replace('\n', ' ')
                
                # Resolve relations
                entities = row['entities']
                relations = row['relations']
                triples = []
                for rel in relations:
                    subj_idx = rel['subject']
                    obj_idx = rel['object']
                    pred = rel['predicate']
                    
                    subj = entities[subj_idx]['surfaceform']
                    obj = entities[obj_idx]['surfaceform']
                    
                    triples.append([subj, pred, obj])
                
                f_source.write(json.dumps(triples, ensure_ascii=False) + '\n')
                f_target.write(text + '\n')

def main():
    base_data_dir = "/data/GraphJudge/data/sredfm_vi"
    output_dirs = [
        "/data/GraphJudge/datasets/sredfm_vi",
        "/data/GraphJudge/data/sredfm_vi"
    ]
    
    splits = ["train", "validation", "test"]
    
    for split in splits:
        arrow_path = os.path.join(base_data_dir, split, "data-00000-of-00001.arrow")
        for out_dir in output_dirs:
            convert_split(arrow_path, out_dir, split)

if __name__ == "__main__":
    main()
