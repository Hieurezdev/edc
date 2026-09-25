import json
import os

def main():
    corpus_path = "corpus.jsonl"
    target_path = "datasets/Qwen3_result_corpus10/test.target"
    
    # Đảm bảo thư mục đích tồn tại
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    print(f"Đang trích xuất văn bản từ {corpus_path} sang {target_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f_in, open(target_path, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            if idx >= 1000:
                break
            data = json.loads(line)
            f_out.write(data["text"].strip().replace("\n", " ") + "\n")
    print("Trích xuất hoàn thành!")

if __name__ == "__main__":
    main()
