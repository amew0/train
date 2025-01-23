from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv

load_dotenv()
import os

run_name = "qwen-7B-r-t4bit-0122_184113"
model_name_or_path = f"out/checkpoints/{run_name}"
model_kwargs = {
    "device_map": "auto",
    # "torch_dtype": torch.float16,
}

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name_or_path, **model_kwargs
)
processor = AutoProcessor.from_pretrained(model_name_or_path)


# push to hub
model.push_to_hub(run_name, token=os.environ["HF_TOKEN_WRITE"])
processor.push_to_hub(run_name, token=os.environ["HF_TOKEN_WRITE"])
