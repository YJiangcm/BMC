from vllm import LLM, SamplingParams
import json
import os
import argparse

def get_json_list2(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--data_path', default="arena_hard.jsonl", type=str)
    parser.add_argument('--tensor_parallel_size', default=1, type=int)
    parser.add_argument('--max_tokens', default=2048, type=int)

    args = parser.parse_args()

    dataset = get_json_list2(args.data_path)

    # Create an LLM.
    llm = LLM(model=f"{args.model_path}", tensor_parallel_size=args.tensor_parallel_size)
    tokenizer = llm.get_tokenizer()

    if 'llama-3' in args.model_path:
        pattern = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    elif 'mistral' in args.model_path or 'zephyr' in args.model_path:
        pattern = (
        "<|user|>\n{}</s>\n<|assistant|>\n"
        )
        stop_token_ids=[tokenizer.eos_token_id]
        
    else:
        raise NameError
    
    # Create a sampling params object.
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=0.0,
                                     max_tokens=args.max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     )

    prompts = []
    for idx in range(len(dataset)):
        prompts.append(pattern.format(dataset[idx]['turns'][0]['content']))

    outputs = llm.generate(prompts, sampling_params)

    with open(f'{args.model_name}.jsonl', 'w', encoding='utf-8') as output_review_file:
        for i in range(len(outputs)):
            output_review_file.write(json.dumps({
                "question_id": dataset[i]['question_id'],
                "answer_id": i,
                "model_id": args.model_name,
                "choices": [{"index": 0, "turns":[{"content": outputs[i].outputs[0].text, "token_len": len(tokenizer.tokenize(outputs[i].outputs[0].text))}]}],
                }) + '\n')