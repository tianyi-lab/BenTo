import json
import os
import time 
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Tuple
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
    'aeslc_10templates',
    'ag_news_subset_10templates',
    'anli_r1_10templates',
    'anli_r2_10templates',
    'anli_r3_10templates',
    'arc_challenge_10templates',
    'arc_easy_10templates',
    'bool_q_10templates',
    'cb_10templates',
    'cnn_dailymail_10templates',
    'cola_10templates',
    'common_gen_10templates',
    'copa_10templates',
    'coqa_10templates',
    'cosmos_qa_10templates',
    'dart_10templates',
    'definite_pronoun_resolution_10templates',
    'drop_10templates',
    'e2e_nlg_10templates',
    'fix_punct_10templates',
    'gigaword_10templates',
    'glue_mrpc_10templates',
    'glue_qqp_10templates',
    'hellaswag_10templates',
    'imdb_reviews_10templates',
    'math_dataset_10templates',
    'mnli_matched_10templates',
    'mnli_mismatched_10templates',
    'multi_news_10templates',
    'multirc_10templates',
    'natural_questions_10templates',
    'openbookqa_10templates',
    'opinion_abstracts_idebate_10templates',
    'opinion_abstracts_rotten_tomatoes_10templates',
    'para_crawl_enes_10templates',
    'paws_wiki_10templates',
    'piqa_10templates',
    'qnli_10templates',
    'quac_10templates',
    'record_10templates',
    'rte_10templates',
    'samsum_10templates',
    'sentiment140_10templates',
    'snli_10templates',
    'squad_v1_10templates',
    'squad_v2_10templates',
    'sst2_10templates',
    'story_cloze_10templates',
    'stsb_10templates',
    'trec_10templates',
    'trivia_qa_10templates',
    'true_case_10templates',
    'web_nlg_en_10templates',
    'wic_10templates',
    'wiki_lingua_english_en_10templates',
    'wmt14_enfr_10templates',
    'wmt16_translate_csen_10templates',
    'wmt16_translate_deen_10templates',
    'wmt16_translate_fien_10templates',
    'wmt16_translate_roen_10templates',
    'wmt16_translate_ruen_10templates',
    'wmt16_translate_tren_10templates',
    'wnli_10templates',
    'word_segment_10templates',
    'wsc_10templates',
    'yelp_polarity_reviews_10templates']


def format_example(train_dataset, idx, include_answer=True, include_question=True):
    prompt = ""
    if include_question:
        prompt += " Input:\n"
        prompt += train_dataset[idx]["inputs"]
        prompt += "\n"
    if include_answer:
        prompt += " Output:\n"
        prompt += train_dataset[idx]["targets"]
        prompt += "\n\n"
    return prompt

def gen_prompt(train_dataset, indices):
    prompt = "You are a helpful AI assistant. Here are some example input-output pairs that you should follow."
    for i in indices:
        prompt += format_example(train_dataset, i)
    return prompt

def prepare_input(tokenizer_left, tokenizer_right, prompts, answers):
    input_tokens = tokenizer_left.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    length = input_tokens['input_ids'].shape[1]
    input_answers = tokenizer_right.batch_encode_plus(answers, return_tensors="pt", padding=True)
    input_ids = torch.cat([input_tokens['input_ids'], input_answers['input_ids']], dim=1).to("cuda")
    attention_mask = torch.cat([input_tokens['attention_mask'], input_answers['attention_mask']], dim=1).to("cuda")
    labels = input_ids.clone()
    labels[:, :length] = -100
    labels = torch.where(labels == tokenizer_right.pad_token_id, -100, labels).to("cuda")
    return {"input_ids":input_ids, "attention_mask":attention_mask, "labels":labels}

def load():
    n_gpus = torch.cuda.device_count()
    tokenizer_left = LlamaTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        cache_dir="model",
        token="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",
        use_fast=False,
        padding_side="left",
    )
    tokenizer_right = LlamaTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        add_bos_token=False,
        cache_dir="model",
        token="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",
        use_fast=False,
        padding_side="right",
    )
    tokenizer_left.pad_token_id = 0 if tokenizer_left.pad_token_id is None else tokenizer_left.pad_token_id
    tokenizer_left.bos_token_id = 1
    tokenizer_right.pad_token_id = 0 if tokenizer_right.pad_token_id is None else tokenizer_right.pad_token_id
    tokenizer_right.bos_token_id = 1
    
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",cache_dir="model",token="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", low_cpu_mem_usage = True, torch_dtype=torch.float16)
    model = tp.tensor_parallel(model, [i for i in range(n_gpus)]) 
    model.eval()

    return model, tokenizer_left, tokenizer_right

def batch_split(prompts, answers, batch_num):
    batch_prompts = []
    batch_answers = []
    mini_batch_prompt = []
    mini_batch_answer = []
    for prompt, answer in zip(prompts, answers):
        mini_batch_prompt.append(prompt)
        mini_batch_answer.append(answer)
        if len(mini_batch_prompt) == batch_num:
            batch_prompts.append(mini_batch_prompt)
            batch_answers.append(mini_batch_answer)
            mini_batch_prompt = []
            mini_batch_answer = []
    if len(mini_batch_prompt) != 0:
        batch_prompts.append(mini_batch_prompt)
        batch_answers.append(mini_batch_answer)
    return batch_prompts, batch_answers

def batch_infer(model, tokenizer_left, tokenizer_right, prompts, answers):
    batch_size = 8
    loss = 0
    length = 0
    batch_prompts, batch_answers = batch_split(prompts, answers, batch_size)
    for batch_input, batch_answers in tqdm(zip(batch_prompts, batch_answers)):
        encode_inputs = prepare_input(tokenizer_left, tokenizer_right, batch_input, batch_answers)
        loss += model(**encode_inputs).loss * len(batch_answers)
        length += len(batch_answers)
    return loss/length

def main(start_task_id: int, example_num: int):
    random.seed(42)
    np.random.seed(42)
    start_task = TASKS[start_task_id]
    model, tokenizer_left, tokenizer_right = load()
    start_time = time.time()
    start_task_dataset = load_dataset("Muennighoff/flan", cache_dir="data", data_files=f"validation/{start_task}_validation.jsonl")["train"].shuffle(seed=42).select(list(range(100)))
    k = args.ntrain
    train_prompt_list = [gen_prompt(start_task_dataset, list(range(k*i, k*(i+1)))) for i in range(example_num)]
    print('starting from task %s' % start_task)
    with torch.no_grad():
        for task in TASKS:
            print('Testing %s ...' % task)
            records = []
            target_task_dataset = load_dataset("Muennighoff/flan", cache_dir="data", data_files=f"validation/{task}_validation.jsonl")["train"].shuffle(seed=42).select(list(range(100)))
            for i in range(len(target_task_dataset)):
                # get prompt and make sure it fits
                prompt_end = format_example(target_task_dataset, i, include_answer=False)
                prompt_answer = format_example(target_task_dataset, i, include_question=False)
                for train_prompt in train_prompt_list:
                    prompt = train_prompt + prompt_end
                    while len(tokenizer_left.tokenize(prompt)) + 1> 2048: # bos token
                        prompt_split = prompt.split("\n\n")
                        prompt_split.pop(1)
                        prompt = '\n\n'.join(prompt_split)                
                    records.append({'prompt':prompt, 'answer':prompt_answer})

            loss = batch_infer(model, tokenizer_left, tokenizer_right, [record['prompt'] for record in records], [record['answer'] for record in records])
            print(f"Eval loss on {task}: {loss}")
    
    end_time = time.time()
    print("total run time %.2f" % (end_time - start_time))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--start_task_id', type=int, default=0)
    parser.add_argument('--example_num', type=int, default=5)
    args = parser.parse_args()
    
    main(args.start_task_id, args.example_num)