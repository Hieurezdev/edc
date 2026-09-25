#!/bin/bash
set -e  # Dừng script nếu có bất kỳ lệnh nào bị lỗi

echo "==========================================="
echo "0. Đang đồng bộ hóa/tạo môi trường ảo uv..."
echo "==========================================="
uv sync

# Đảm bảo các thư mục đầu ra tồn tại
mkdir -p datasets/Qwen3_result_corpus10/Iteration1
mkdir -p datasets/Qwen3_result_corpus10/Graph_Iteration1

echo "==========================================="
echo "1. Đang trích xuất văn bản từ corpus.jsonl..."
echo "==========================================="
uv run python ./datasets/prepare_corpus.py

echo "==========================================="
echo "2. Chạy Entity Extraction & Denoising..."
echo "==========================================="
uv run python ./chat/run_chatgpt_entity.py

echo "==========================================="
echo "3. Chạy Triple Generation..."
echo "==========================================="
uv run python ./chat/run_chatgpt_triple.py

echo "==========================================="
echo "4. Định dạng đồ thị sang CSV..."
echo "==========================================="
uv run python ./datasets/prepare_KGCom.py \
    --task format-graphs \
    --dataset_path ./datasets/Qwen3_result_corpus10/ \
    --iteration 1

echo "==========================================="
echo "5. Chạy Batch Inference với mô hình finetuned..."
echo "==========================================="
cd ./graph_judger
uv run python lora_infer_batch.py \
    --finput ../datasets/Qwen3_result_corpus10/Graph_Iteration1/test_instructions_llama2_7b_itr1.csv \
    --foutput ../datasets/Qwen3_result_corpus10/Graph_Iteration1/pred_instructions_context_llama2_7b_itr1.csv
cd ..

echo "==========================================="
echo "6. Lọc các bộ ba sai (Triple Filtering)..."
echo "==========================================="
uv run python ./datasets/prepare_KGCom.py \
    --task filter-graphs \
    --dataset_path ./datasets/Qwen3_result_corpus10/ \
    --iteration 1 \
    --pred_model llama2_7b

echo "==========================================="
echo "7. Sao chép toàn bộ kết quả sang Google Drive (/content/drive/MyDrive/)..."
echo "==========================================="
mkdir -p /content/drive/MyDrive/Qwen3_result_corpus10
cp -r datasets/Qwen3_result_corpus10/* /content/drive/MyDrive/Qwen3_result_corpus10/

echo "==========================================="
echo "Pipeline hoàn thành thành công!"
echo "Kết quả bộ ba cuối cùng đã được lưu tại: /content/drive/MyDrive/Qwen3_result_corpus10/Graph_Iteration1/test_generated_graphs_llama2_7b_final.txt"
echo "==========================================="








========================= FINAL SUMMARY GraphJudge =========================

Split: test
  Accuracy: 80.0937%
  Precision: 78.8279%
  Recall: 82.4365%
  F1-Score: 80.5918%





========================= FINAL SUMMARY LLM =========================
Split: test
  Accuracy: 54.4075%
  Precision: 57.6523%
  Recall: 34.1368%
  F1-Score: 42.8823%
================================================================