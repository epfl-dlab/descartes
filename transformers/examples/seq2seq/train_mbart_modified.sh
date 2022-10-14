python run_summarization.py \
    --learning_rate=3e-5 \
    --do_eval \
    --do_train \
    --evaluation_strategy="steps" \
    --freeze_encoder \
    --freeze_embeds \
    --max_source_length 512 \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --num_train_epochs 5 \
    --dataloader_num_workers 1 \
    --tgt_lang en_XX \
    --src_lang en_XX \
    --save_steps 100000 \
    --eval_steps 100000 \
    --tokenizer_name facebook/mbart-large-cc25 \
    --model_name_or_path facebook/mbart-large-cc25 \
    --languages en_XX,cs_CZ,fr_XX,it_IT,fi_FI,de_DE,nl_XX,ro_RO,ru_RU,ja_XX \
    "$@"
