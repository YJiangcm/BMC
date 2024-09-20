set -ex

cd eval

PROMPT_TYPE="platypus_fs"
MODEL_NAME_OR_PATH="../../../../outputs/llama-2-7b-base-dpo-bmc-qa"


OUTPUT_DIR="../../results/eval"
SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="gsm8k,math-oai,mawps,tabmwp"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite