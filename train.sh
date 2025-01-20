NNODES=1
NODE_RANK=0
# ADDR=$(hostname -I | awk '{print $1}')
# NUM_GPUS=$(nvidia-smi -L | wc -l)
# PORT=29500
ADDR=172.18.11.3
NUM_GPUS=4
PORT=$(( ( RANDOM % 55000 )  + 10000 )) # Random port between 10000 and 65000

echo NNODES: $NNODES
echo NODE_RANK: $NODE_RANK
echo ADDR: $ADDR
echo NUM_GPUS: $NUM_GPUS
echo PORT: $PORT

DATA_PATH=/home/kunet.ae/ku5001069/j/generator/data/p-train.json
MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
if [ -z "$1" ]; then
    RUN_NAME="qwen-2B-p-t4bit-$(date +%m%d_%H%M%S)"
else
    RUN_NAME="qwen-2B-p-t4bit-$1"
fi
# RUN_NAME="put the checkpoint name here"
OUTPUT_DIR=/dpc/kunf0097/out/checkpoints/$RUN_NAME

echo "MODEL_NAME: $MODEL_NAME"
echo "RUN_NAME: $RUN_NAME"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "DATA_PATH: $DATA_PATH"

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${ADDR}" \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${PORT}" \
    train.py \
    --deepspeed config/zero3.json \
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
    --load_in_4bit True \
    # --eval_strategy "steps" \
    # --eval_steps 0.05 \
    # --batch_eval_metrics \