import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import os
import json
from tqdm import tqdm

### VARS
dataset_name = "/home/kunet.ae/ku5001069/j/generator/data/p-test.json"
for_r = "rec" in dataset_name
video_dir = "/dpc/kunf0097/data/cchw"
# run_name = "qwen-7B-r-t4bit-0122_184113"
# model_name_or_path = f"/dpc/kunf0097/.cache/huggingface/hub/{run_name}"
model_name_or_path = f"Qwen/Qwen2-VL-7B-Instruct"
output_name = f"out/p-test-{model_name_or_path.split('/')[-1]}.json"  # rename it better
###

# print vars
print(f"\n\nDataset: {dataset_name}")
print(f"Model: {model_name_or_path}")
print(f"Output: {output_name}\n\n")


model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

processor = AutoProcessor.from_pretrained(model_name_or_path, max_pixels=(300 * 300))
with open(dataset_name, "r") as f:
    examples = json.load(f)


def formatting_func(example):
    msgs = example["messages"]
    msgs[1]["content"] = json.loads(msgs[1]["content"])
    if not for_r:
        msgs[1]["content"][0]["video"] = f"{video_dir}/{msgs[1]['content'][0]['video']}"
    return msgs


records = []
for i, example in tqdm(enumerate(examples), total=len(examples)):

    messages = formatting_func(example)
    text = processor.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )
    if for_r:
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
    else:
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=1000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    final_output = output_text[0]
    video_file_name = (
        messages[1]["content"][0]["video"] if not for_r else example["video_file_name"]
    )
    record = {
        "generated": final_output,
        "expected": messages[2]["content"],
        "video_file_name": video_file_name,
    }
    records.append(record)

    with open(output_name, "w") as f:
        json.dump(records, f, indent=2)

    if i == 0:
        print("\n\n")
        print(record)

print("Done!!")
print(f"Output saved to {output_name}")
