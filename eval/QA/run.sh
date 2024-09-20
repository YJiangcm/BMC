

model_path="../../../outputs/llama-2-7b-base-dpo-bmc-qa"


cd ./ECQA/
python eval.py \
    --model $model_path \
    --n_shot 0 \
    --result_file ecqa-0_shot.jsonl
cd ..


cd ./QASC/
python eval.py \
    --model $model_path \
    --n_shot 0 \
    --result_file qasc-0_shot.jsonl
cd ..


cd ./OpenbookQA/
python eval.py \
    --model $model_path \
    --n_shot 0 \
    --result_file openbookqa-0_shot.jsonl
cd ..


cd ./ARC/
python eval.py \
    --model $model_path \
    --n_shot 0 \
    --result_file arc-0_shot.jsonl
cd ..