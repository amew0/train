import os
import json


def find_assistant_content_sublist_indexes(l):
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


def collate_fn(examples, processor):
    # Cut from the original collate_fn
    # mask the padding token & the video pad token (check the model config file)
    labels = batch["input_ids"].clone()
    video_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)

    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == video_pad_id] = -100

    batch["labels"] = labels
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
