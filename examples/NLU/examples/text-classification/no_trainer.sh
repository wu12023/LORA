export TASK_NAME=cola

python run_glue_no_trainer.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --max_length 512 \
  --per_device_train_batch_size 32 \
  --learning_rate 4e-4 \
  --num_train_epochs 20 \
  --output_dir /tmp/$TASK_NAME\
  --weight_decay 0.1 \
  --seed 0 \