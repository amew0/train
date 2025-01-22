DATA_PATH=/home/ubuntu/train/out/rec-train.json
MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct
if [ -z "$1" ]; then
    RUN_NAME="qwen-7B-p-t4bit-$(date +%m%d_%H%M%S)"
else
    RUN_NAME="qwen-7B-p-t4bit-$1"
fi
# RUN_NAME="put the checkpoint name here"
OUTPUT_DIR=/home/ubuntu/train/out/checkpoints/$RUN_NAME

echo "MODEL_NAME: $MODEL_NAME"
echo "RUN_NAME: $RUN_NAME"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "DATA_PATH: $DATA_PATH"

# export TRITON_CACHE_DIR="/dpc/kunf0097/.cache/triton"
# accelerate test --config-file="config/d.yaml"

accelerate launch \
    --config-file="config/d.yaml" \
    train_acc.py \
    --dataset_name ${DATA_PATH} \
    --model_name_or_path ${MODEL_NAME} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --run_name ${RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --use_peft \
    --report_to wandb \
    --num_train_epochs 10 \
    --fp16 True \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --load_in_4bit True