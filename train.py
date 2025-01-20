import torch
import pathlib
import json
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

import torch.distributed as dist
import os


from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from utils import *

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"{dist.get_rank()}: ", *args)
    else:
        print(*args)

# collate_fn
# assist from https://github.com/zhangfaen/finetune-Qwen2-VL/blob/main/finetune.py
def collate_fn(examples):
    # rank0_print(len(examples))
    for_r = "rec" in script_args.dataset_name
    messages = [e["messages"] for e in examples]
    texts = [processor.apply_chat_template(m, tokenize=False) for m in messages]
    if for_r:
        raise NotImplementedError("rec dataset not supported yet")
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


if __name__ == "__main__":
    print_gpu_utilization()

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    training_args.gradient_checkpointing_kwargs = dict(
        use_reentrant=False
    )  # for distributed training
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    model_config.lora_target_modules = [
        "down_proj",
        "k_proj",
        "up_proj",
        "q_proj",
        "v_proj",
        "fc2",
        "qkv",
        "gate_proj",
        "o_proj",
        "fc1",
        "proj",
        # "0",
        # "2",
    ]
    # TODO: model_config.lora_target_modules = find_all_linear_names(model)

    rank0_print(f"Script Args: {vars(script_args)}\n\n")
    rank0_print(f"Training Args: {vars(training_args)}\n\n")
    rank0_print(f"Model Config: {vars(model_config)}\n\n")

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        # device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        padding_side="right",  # needed for training qwen2-vl
        # max_pixels=(200 * 200), # this doesn't seem to be used!!!
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs,
    )

    print_gpu_utilization()

    train_dataset = load_dataset(
        "json", data_files=script_args.dataset_name, split="train"
    )
    if os.path.exists(training_args.output_dir):
        train_dataset = reorder_dataset(train_dataset, training_args.output_dir)

    from torch.utils.data import SequentialSampler

    class SFTTrainerNoShuffle(SFTTrainer):
        def _get_train_sampler(self):
            return SequentialSampler(self.train_dataset)  # prevents shuffling

    trainer = SFTTrainerNoShuffle(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        peft_config=get_peft_config(model_config),
        processing_class=processor.tokenizer,
    )
    torch.cuda.empty_cache()

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)  # added!

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
            
    rank0_print("Training completed successfully!")
