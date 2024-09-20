from vllm import LLM, SamplingParams
import json
import argparse

def get_json_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as fcc_file:
        json_list = json.load(fcc_file)
    return json_list
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--model_name', default=None, type=str)
    parser.add_argument('--data_path', default="alpaca_eval.json", type=str)
    parser.add_argument('--tensor_parallel_size', default=1, type=int)
    parser.add_argument('--max_tokens', default=2048, type=int)

    args = parser.parse_args()

    dataset = get_json_list(args.data_path)

    # Create an LLM.
    llm = LLM(model=f"{args.model_path}", tensor_parallel_size=args.tensor_parallel_size)
    tokenizer = llm.get_tokenizer()

    if 'llama-3' in args.model_path:
        pattern = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    elif 'mistral' in args.model_path:
        pattern = (
        "<|user|>\n{}</s>\n<|assistant|>\n"
        )
        stop_token_ids=[tokenizer.eos_token_id]
        
    else:
        raise NameError
    
    # Create a sampling params object.
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=0.9,
                                     max_tokens=args.max_tokens,
                                     stop_token_ids=stop_token_ids,
                                     )

    prompts = []
    for idx in range(len(dataset)):
        prompts.append(pattern.format(dataset[idx]['instruction']))

    outputs = llm.generate(prompts, sampling_params)

    saved_output = []
    for idx in range(len(outputs)):
        saved_output.append({
                            'dataset': dataset[idx]['dataset'],
                            'instruction': dataset[idx]['instruction'], 
                            'output': outputs[idx].outputs[0].text,
                            'generator': args.model_name
                             })
        
        
    json_str = json.dumps(saved_output, indent=4)
    with open(f"model_outputs_{args.model_name}.json", mode='w', encoding='utf-8') as json_file:
        json_file.write(json_str)