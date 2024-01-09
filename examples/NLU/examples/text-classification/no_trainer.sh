export TASK_NAME=cola

python run_glue_no_trainer.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 4e-4 \
  --num_train_epochs 80 \
  --output_dir /home/wuyujia/LoRA/examples/NLU/examples/text-classification/temp/$TASK_NAME/ \
  --weight_decay 0.1\
