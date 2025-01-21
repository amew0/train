import evaluate
from datetime import datetime
from tqdm import tqdm
import json

bert = evaluate.load("bertscore")
gleu = evaluate.load("google_bleu")


class MetricTracker:
    def __init__(self):
        self.bert_p = []
        self.bert_r = []
        self.bert_f1 = []
        self.gleu = []

    def update(self, predicted_deto, label_deto):
        # Calculate scores for this batch
        bert_score = bert.compute(
            predictions=predicted_deto,
            references=label_deto,
            model_type="microsoft/deberta-v3-small",
        )
        gleu_score = gleu.compute(predictions=predicted_deto, references=label_deto)

        self.bert_p.extend(bert_score["precision"])
        self.bert_r.extend(bert_score["recall"])
        self.bert_f1.extend(bert_score["f1"])
        self.gleu.append(
            gleu_score["google_bleu"]
        )  # gleu avg batch single score returned

    def compute(self, log_for="eval"):
        # Aggregate scores across batches
        mean = lambda l: "{:.3f}".format(sum(l) / len(l)) if l else "0.0"
        result = {
            f"{log_for}/bert_p": mean(self.bert_p),
            f"{log_for}/bert_r": mean(self.bert_r),
            f"{log_for}/bert_f1": mean(self.bert_f1),
            f"{log_for}/gleu": mean(self.gleu),
        }

        # Reset batch statistics
        self.bert_p = []
        self.bert_r = []
        self.bert_f1 = []
        self.gleu = []

        return result


metric_tracker = MetricTracker()

files_to_evaluate = [
    # "out/p-test-qwen-2B-p-t4bit-0120_141519.json",
    # "out/p-test-Qwen2-VL-2B-Instruct.json",
    # rec
    "out/rec-test-Qwen2-VL-2B-Instruct.json",
    "out/rec-test-qwen-2B-r-t4bit-0121_153914.json",
    "out/rec-test-qwen-2B-r-t4bit-0121_160123.json",
]


save_path = f"out/eval-{datetime.now().strftime('%m%d_%H%M%S')}.json"
with open(save_path, "w") as f:
    json.dump([], f)

results = []

for i, file in tqdm(enumerate(files_to_evaluate), total=len(files_to_evaluate)):
    print(f"File {i+1}/{len(files_to_evaluate)}: {file}")
    with open(file) as f:
        examples = json.load(f)
    preds = [ex["generated"] for ex in examples]
    labels = [ex["expected"] for ex in examples]

    metric_tracker.update(preds, labels)

    result = metric_tracker.compute(log_for="eval")
    print(result)
    results.append({"file": file, **result})

    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

print(f"Results saved to {save_path}")
