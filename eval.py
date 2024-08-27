from datasets import load_dataset
import argparse
import transformers
import accelerate
from tasks import TASKS
from prompts import gen_prompt, format_example, format_example_flan
import os
from inference import batch_infer
from metric import compute_metric
import torch
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="mmlu", choices=["mmlu", "flan", "bbh", "agieval"])
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use_remote_data", action="store_true")
    parser.add_argument("--is_reduced", action="store_true")
    parser.add_argument("--example_num", type=int, default=5)
    parser.add_argument("--eval_times", type=int, default=5)
    parser.add_argument("--output_filename", type=str, default="results.json")
    parser.add_argument("--model", type=str, default="microsoft/phi-2")
    args = parser.parse_args()
    
    reduce_string = ""
    if args.is_reduced:
        reduce_string = "_reduced"
    benchmark = args.task + reduce_string
    if args.use_remote_data:
        load_path = "cindermond/bento"
        cache_dir = args.data_folder
    else:
        load_path = args.data_folder
        cache_dir = None
    
    try:
        token = os.environ["HF_TOKEN"]
    except KeyError:
        token = None

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", token=token, low_cpu_mem_usage=True, torch_dtype=torch.float16, cache_dir=args.cache_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, padding_side="left", use_fast=False, cache_dir=args.cache_dir, token=token)
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    model.eval()
    run_results = {}
    for task in TASKS[benchmark]:
        records = []
        train_dataset = load_dataset(load_path, data_files = benchmark + "/" + task + "/dev.jsonl", cache_dir=cache_dir)["train"]
        test_dataset = load_dataset(load_path, data_files = benchmark + "/" + task + "/test.jsonl", cache_dir=cache_dir)["train"]
        subject = None
        if args.task == "mmlu":
            subject = task
        if args.task == "flan":
            met = []
        prompts = gen_prompt(benchmark, train_dataset, args.example_num, args.eval_times, subject=subject)
        for data in test_dataset:
            prompt_end = format_example(benchmark, data)
            extended_prompts = [prompt + prompt_end for prompt in prompts]
            for i in range(len(extended_prompts)):
                prompt = extended_prompts[i]
                bad_exp = False
                while len(tokenizer.tokenize(prompt)) + 1> 2048:
                    prompt_split = prompt.split("\n\n")
                    if len(prompt_split) == 1:
                        bad_exp = True
                        break
                    prompt_split.pop(1)
                    prompt = '\n\n'.join(prompt_split)
                if not bad_exp:
                    if args.task == "mmlu":
                        answer = data["answer"]
                    if args.task == "flan":
                        answer = format_example_flan(data, include_question=False, include_answer=True)
                    if args.task == "bbh":
                        answer = data["target"]
                    if args.task == "agieval":
                        answer = data["label"]
                    records.append({"prompt": prompt, "answer": answer})
        pred_answers = batch_infer(benchmark, model, tokenizer, [record['prompt'] for record in records], [record['answer'] for record in records])
        if args.task == "flan":
            print(f"Eval loss on {task}: {pred_answers}")
            met.append(pred_answers)
        else:
            gold_answers = [record['answer'] for record in records]
            run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
    if args.task != "flan":
        with open(args.output_filename, 'w') as f:
            json.dump(run_results, f, ensure_ascii=False, indent=2)
        compute_metric(benchmark, args.output_filename) 
    else:
        flan_metric = sum([exp(loss) for loss in met])/len(met)       
        print(f"Total loss on {benchmark}: {flan_metric}")