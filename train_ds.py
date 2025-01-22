import os
import torch
import pathlib
from utils import (
    collate_helper,
    find_all_linear_names,
    print_gpu_utilization,
    reorder_dataset,
)
from datasets import load_dataset
import torch.distributed as dist
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
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
from torch.utils.data import SequentialSampler
import deepspeed

class SFTTrainerNoShuffle(SFTTrainer):
    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)  # prevents shuffling

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"{dist.get_rank()}: ", *args)
    else:
        print(*args)


if __name__ == "__main__":
    print_gpu_utilization()

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    # training_args.gradient_checkpointing_kwargs = dict(
    #     use_reentrant=False
    # )  # for distributed training
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    model_config.lora_target_modules = find_all_linear_names()

    rank0_print(f"Script Args: {vars(script_args)}\n\n")
    rank0_print(f"Training Args: {vars(training_args)}\n\n")
    rank0_print(f"Model Config: {vars(model_config)}\n\n")

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    # quantization_config = get_quantization_config(model_config)
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

    collate_fn = lambda examples: collate_helper(
        examples, processor, for_r="rec" in script_args.dataset_name
    )

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
