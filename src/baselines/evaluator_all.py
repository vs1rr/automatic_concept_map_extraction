import os
import json

def compute_average_metrics(folder_path):
    total_precision_meteor = 0
    total_recall_meteor = 0
    total_f1_meteor = 0
    total_precision_rouge2 = 0
    total_recall_rouge2 = 0
    total_f1_rouge2 = 0
    count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)

                meteor = data.get('meteor', {})
                rouge2 = data.get('rouge-2', {})

                precision_meteor = meteor.get('precision', 0)
                recall_meteor = meteor.get('recall', 0)
                f1_meteor = meteor.get('f1', 0)

                precision_rouge2 = rouge2.get('precision', 0)
                recall_rouge2 = rouge2.get('recall', 0)
                f1_rouge2 = rouge2.get('f1', 0)

                total_precision_meteor += precision_meteor
                total_recall_meteor += recall_meteor
                total_f1_meteor += f1_meteor

                total_precision_rouge2 += precision_rouge2
                total_recall_rouge2 += recall_rouge2
                total_f1_rouge2 += f1_rouge2

                count += 1

    average_metrics = {
        "meteor": {
            "precision": total_precision_meteor / count if count > 0 else 0,
            "recall": total_recall_meteor / count if count > 0 else 0,
            "f1": total_f1_meteor / count if count > 0 else 0
        },
        "rouge-2": {
            "precision": total_precision_rouge2 / count if count > 0 else 0,
            "recall": total_recall_rouge2 / count if count > 0 else 0,
            "f1": total_f1_rouge2 / count if count > 0 else 0
        }
    }

    return average_metrics

def main():
    folder_path = "./experiments_emnlp/baselines/zero_baseline/results-zero-train"
    average_metrics = compute_average_metrics(folder_path)

    with open("./experiments_emnlp/baselines/zero_baseline/results-zero-train/average_metrics_zero.json", "w") as outfile:
        json.dump(average_metrics, outfile, indent=4)

    print("Average metrics computed and saved successfully.")

if __name__ == "__main__":
    main()
