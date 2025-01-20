import torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model_kwargs = {
    "device_map": "cuda:0",
    "torch_dtype": torch.float16,
}
base_model_path = "Qwen/Qwen2-VL-2B-Instruct"
model_name_or_path = (
    "/dpc/kunf0097/out/checkpoints/qwen-2B-p-t4bit-0118_014847"
)

model = Qwen2VLForConditionalGeneration.from_pretrained(base_model_path, **model_kwargs)
processor = AutoProcessor.from_pretrained(base_model_path)


peft_model = PeftModel.from_pretrained(model, model_name_or_path)
merged_model = peft_model.merge_and_unload()

save_path = "/dpc/kunf0097/.cache/huggingface/hub/qwen-2B-p-t4bit-0118_014847"
merged_model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print("Saved to", save_path)