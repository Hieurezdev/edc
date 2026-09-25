import os
import sys
import argparse
import matplotlib.pyplot as plt

# Import the helper function from datasets/prepare_KGCom.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare_KGCom import extract_list_from_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="GPT4o_mini_result_corpus10", help="Dataset folder name")
    parser.add_argument("--iteration", type=int, default=1, help="Graph iteration number")
    args = parser.parse_args()

    before_file = f'datasets/{args.folder}/Graph_Iteration{args.iteration}/test_generated_graphs.txt'
    after_file = f'datasets/{args.folder}/Graph_Iteration{args.iteration}/test_generated_graphs_llama2_7b_final.txt'
    output_image = f'datasets/{args.folder}/Graph_Iteration{args.iteration}/triple_comparison.png'

    if not os.path.exists(before_file):
        print(f"Error: Original graphs file not found at: {before_file}")
        sys.exit(1)
    if not os.path.exists(after_file):
        print(f"Error: Filtered graphs file not found at: {after_file}")
        sys.exit(1)

    before_counts = []
    with open(before_file, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            triples = extract_list_from_text(l.strip())
            before_counts.append(len(triples))

    after_counts = []
    with open(after_file, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            triples = extract_list_from_text(l.strip())
            # Filter out fallback empty list containing ['none', 'none', 'none']
            if len(triples) == 1 and triples[0] == ['none', 'none', 'none']:
                after_counts.append(0)
            else:
                after_counts.append(len(triples))

    total_before = sum(before_counts)
    total_after = sum(after_counts)
    retained_ratio = (total_after / total_before) * 100 if total_before > 0 else 0

    print(f"STATS:before={total_before},after={total_after},ratio={retained_ratio:.2f}%")
    print(f"BEFORE_LIST:{before_counts}")
    print(f"AFTER_LIST:{after_counts}")

    # Plotting the comparison chart
    samples = [f"S{i+1}" for i in range(len(before_counts))]

    plt.figure(figsize=(10, 6))

    # Modern styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = range(len(samples))

    # Harmonious HSL-like colors (Sleek modern theme)
    color_before = '#5c7cfa' # Indigo/blue
    color_after = '#37b24d'  # Green

    rects1 = ax.bar(index, before_counts, bar_width, label='Trước khi lọc (Original)', color=color_before, alpha=0.85)
    rects2 = ax.bar([i + bar_width for i in index], after_counts, bar_width, label='Sau khi lọc (Filtered)', color=color_after, alpha=0.85)

    ax.set_xlabel('Mẫu tài liệu (Samples)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Số lượng Triple (Count)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'So sánh số lượng Triples trước và sau khi lọc ({args.folder})\n(Giữ lại: {total_after}/{total_before} triples ~ {retained_ratio:.2f}%)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(samples, fontsize=10)
    ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=11)

    # Adding value labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Chart successfully saved to {output_image}")

if __name__ == "__main__":
    main()
