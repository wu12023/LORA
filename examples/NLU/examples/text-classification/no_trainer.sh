export TASK_NAME=sst2

CUDA_VISIBLE_DEVICES=4 python -u run_glue_no_trainer.py \
  --model_name_or_path roberta-base \
  --wandb_name test_seed_8765\
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-4 \
  --num_train_epochs 60 \
  --output_dir /home/wuyujia/LoRA/examples/NLU/examples/text-classification/temp/$TASK_NAME/ \
  --weight_decay 0.1\
  --seed 0\
  --apply_lora \
  --lora_r 8 \
  --lora_alpha 8 \
  --warmup_ratio 0.06\ > results/${TASK_NAME}_seed_${seed}_cuda_4.log 2>&1