import torch
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


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"{dist.get_rank()}: ", *args)
    else:
        print(*args)


"""def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
"""


def collate_fn(examples):
    # TODO: DataCollatorForCompletionOnly in the next life
    for_r = "rec" in script_args.dataset_name
    texts = [
        processor.apply_chat_template(e, tokenize=False) for e in examples
    ]
    if for_r:
        raise NotImplementedError("rec dataset not supported yet")
    else:
        image_inputs, video_inputs = process_vision_info([e['messages'] for e in examples])

        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    # mask the padding token & the video pad token (check the model config file)
    labels = batch["input_ids"].clone()
    video_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)

    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == video_pad_id] = -100

    batch["labels"] = labels
    return batch


if __name__ == "__main__":

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
        # quantization_config=quantization_config,
        torch_dtype=torch_dtype,
    )
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_config.model_name_or_path,
        **model_kwargs,
    )

    if os.path.exists(script_args.dataset_name):
        dataset = load_dataset("json", data_files=script_args.dataset_name)
    else:
        dataset = load_dataset(script_args.dataset_name)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset[script_args.dataset_train_split],
        peft_config=get_peft_config(model_config),
    )
    torch.cuda.empty_cache()

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)  # added!

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        if trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)
            
    rank0_print("Training completed successfully!")
