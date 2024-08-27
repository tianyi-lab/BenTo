import json
import os
import time 
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM
import tensor_parallel as tp
import accelerate
import random
import numpy as np
from math import ceil



TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

choices = ["A", "B", "C", "D"]

def compute_metric(output_filename):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        print("ACC-%s: %.4f" % (task, acc/len(gold_answers)))
        total_acc += acc
        total_num += len(gold_answers)
    print("ACC-all: %.4f" % (total_acc/total_num))


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def gen_prompt_sub(subject):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    return prompt

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def load(ckpt_dir, model_type):
    n_gpus = torch.cuda.device_count()
    tokenizer = LlamaTokenizer.from_pretrained(
        "meta-llama/Llama-2-13b-hf",
        cache_dir="model",
        use_fast=False,
        padding_side="left",
    )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1
    
    if model_type == 'llama':
        # we use tensor parallel for loading llama
        model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf",cache_dir="model", low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)]) 
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map = 'balanced_low_0', torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1,pad_token_id=0)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers

def main(ckpt_dir: str, param_size: str, model_type: str, start_task_id: int, example_num: int):
    random.seed(42)
    np.random.seed(42)
    run_results = {}
    start_task = TASKS[start_task_id]
    output_filename = 'log/normal/raw_results/run_results_%s_%sb_%s_%d.json' % (model_type, param_size, start_task, example_num)
    
    model, tokenizer = load(ckpt_dir, model_type)
    start_time = time.time()
    dev_df = pd.read_csv(os.path.join(args.data_dir, "test", start_task + "_test.csv"), header=None).sample(frac=1).reset_index(drop=True)[:args.ntrain*example_num]
    k = args.ntrain
    train_prompt_list = [gen_prompt(dev_df[(i*args.ntrain):(i*args.ntrain+args.ntrain)], start_task, k) for i in range(ceil(len(dev_df.index)/args.ntrain))]
    print('starting from task %s' % start_task)
    for task in TASKS:
        print('Testing %s ...' % task)
        records = []
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        for i in range(test_df.shape[0]):
            # get prompt and make sure it fits
            prompt_end = format_example(test_df, i, include_answer=False)
            #prompt_sub = gen_prompt_sub(task)
            prompt_sub = ""
            label = test_df.iloc[i, test_df.shape[1]-1]
            for train_prompt in train_prompt_list:
                prompt = train_prompt + prompt_sub + prompt_end
                while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                    prompt_split = prompt.split("\n\n")
                    prompt_split.pop(1)
                    prompt = '\n\n'.join(prompt_split)                
                records.append({'prompt':prompt, 'answer':label})


        pred_answers = batch_infer(model, tokenizer, [record['prompt'] for record in records])
        gold_answers = [record['answer'] for record in records]
        run_results[task] = {'pred_answers':pred_answers, 'gold_answers':gold_answers}
    with open(output_filename, 'w') as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    
    compute_metric(output_filename)
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, default='model/models--meta-llama--Llama-2-13b-hf')
    parser.add_argument('--param_size', type=str, default='13')
    parser.add_argument('--model_type', type=str, default='llama')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--start_task_id', type=int, default=0)
    parser.add_argument('--example_num', type=int, default=10)
    args = parser.parse_args()
    
    main(args.ckpt_dir, args.param_size, args.model_type, args.start_task_id, args.example_num)
