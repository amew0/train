import os
import json
import torch
from qwen_vl_utils import process_vision_info


def find_assistant_content_sublist_indexes(l):
    # from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py
    """
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    """
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091 and l[i + 2] == 198:
            start_indexes.append(i + 3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i + 3, len(l) - 1):
                if l[j] == 151645 and l[j + 1] == 198:
                    end_indexes.append(
                        j + 2
                    )  # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


def collate_helper(examples, processor, for_r=False):
    messages = [e["messages"] for e in examples]
    texts = [processor.apply_chat_template(m, tokenize=False) for m in messages]
    if for_r:
        batch = processor(text=texts, padding=True, return_tensors="pt")
    else:
        for i in range(len(messages)):
            messages[i][1]["content"] = json.loads(messages[i][1]["content"])

        image_inputs, video_inputs = process_vision_info(messages)

        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    input_ids_lists = batch["input_ids"].tolist()
    assert len(messages) == len(input_ids_lists)

    # TODO: just use DataCollatorForCompletionOnly
    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0] : begin_end_indexs[1]] = ids_list[
                begin_end_indexs[0] : begin_end_indexs[1]
            ]
        labels_list.append(label_ids)
    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    batch["labels"] = labels_ids
    return batch


from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def get_last_checkpoint(chpt_dir):
    checkpoints = [d for d in os.listdir(chpt_dir) if d.startswith("checkpoint-")]
    last_checkpoint = None
    if checkpoints:
        last_checkpoint = os.path.join(
            chpt_dir,
            max(checkpoints, key=lambda cp: int(cp.split("-")[-1])),
        )
    return last_checkpoint


def get_start_index(chpt_dir, n_examples) -> int:
    last_checkpoint = get_last_checkpoint(chpt_dir)
    if last_checkpoint == None:
        print(
            "Dataset: No checkpoint found. Starting from the beginning. This is weird since the checkpoint directory exists already, but it does not have any checkpoints."
        )
        return 0
    with open(os.path.join(last_checkpoint, "trainer_state.json"), "r") as f:
        trainer_state = json.load(f)

    start_index = (n_examples * trainer_state["epoch"]) % n_examples
    return int(start_index)


def reorder_dataset(dataset, chpt_dir):
    from datasets import concatenate_datasets

    # TODO: Verify if it's resuming correctly, there is a bug multiplying the index by 2
    start_index = get_start_index(chpt_dir, len(dataset))
    print(f"Dataset: Resuming from index {start_index}/{len(dataset)}.")

    # Split the dataset into two parts: before and after the start index
    dataset_part1 = dataset.select(range(start_index, len(dataset)))
    dataset_part2 = dataset.select(range(start_index))

    # Concatenate the two parts
    reordered_dataset = concatenate_datasets([dataset_part1, dataset_part2])
    return reordered_dataset


def calculate_parameters_per_module(model, target_module_names):
    parameters_per_module = {name: 0 for name in target_module_names}
    for name, module in model.named_modules():
        # Check if the name matches one of the target names
        for target_name in target_module_names:
            if target_name in name:
                # Add the parameters for this module to the corresponding key
                parameters_per_module[target_name] += sum(
                    p.numel() for p in module.parameters()
                )
    return parameters_per_module


# # Example usage
# target_module_names = ["down_proj", "k_proj", "up_proj", "q_proj", "v_proj"]
# parameters_per_module = calculate_parameters_per_module(model, target_module_names)

# # Print results
# for module_name, param_count in parameters_per_module.items():
#     print(f"#P's {module_name}: {param_count/1e6}m")


def find_all_linear_names():
    # TODO: model_config.lora_target_modules = find_all_linear_names(model)
    return [
        # # llm
        "down_proj",
        "k_proj",
        "up_proj",
        "q_proj",
        "v_proj",
        "gate_proj",
        "o_proj",
        # # vision
        # "fc1",
        # "fc2",
        # "qkv",
        # "proj",
        # # merge
        # "0",
        # "2",
    ]
