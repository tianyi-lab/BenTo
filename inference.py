from tqdm import tqdm
import torch


def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')
    return input_tokens

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

def batch_infer_multiple_choice(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers

def prepare_input_flan(tokenizer, prompts, answers):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    length = input_tokens['input_ids'].shape[1]
    tokenizer.padding_side = "right"
    input_answers = tokenizer.batch_encode_plus(answers, return_tensors="pt", padding=True)
    input_ids = torch.cat([input_tokens['input_ids'], input_answers['input_ids']], dim=1).to("cuda")
    attention_mask = torch.cat([input_tokens['attention_mask'], input_answers['attention_mask']], dim=1).to("cuda")
    labels = input_ids.clone()
    labels[:, :length] = -100
    labels = torch.where(labels == tokenizer.pad_token_id, -100, labels).to("cuda")
    tokenizer.padding_side = "left"
    return {"input_ids":input_ids, "attention_mask":attention_mask, "labels":labels}

def batch_split_flan(prompts, answers, batch_num):
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

def batch_infer_flan(model, tokenizer, prompts, answers):
    batch_size = 8
    loss = 0
    length = 0
    batch_prompts, batch_answers = batch_split_flan(prompts, answers, batch_size)
    for batch_input, batch_answers in tqdm(zip(batch_prompts, batch_answers)):
        encode_inputs = prepare_input_flan(tokenizer, batch_input, batch_answers)
        loss += model(**encode_inputs).loss * len(batch_answers)
        length += len(batch_answers)
    return loss/length


def batch_infer_bbh(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=512)
        batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        final_answers = []
        for answer in batch_answers:
            try:
                parsed_answer = answer.split("So the answer is ")[5].split(".")[0]
                final_answers.append(parsed_answer)
            except:
                final_answers.append("")
        answers.extend(final_answers)
    return answers

def batch_infer(benchmark, model, tokenizer, prompts, answers=None):
    if 'agieval' in benchmark or 'mmlu' in benchmark:
        return batch_infer_multiple_choice(model, tokenizer, prompts)
    elif 'flan' in benchmark:
        return batch_infer_flan(model, tokenizer, prompts, answers)
    elif 'bbh' in benchmark:
        return batch_infer_bbh(model, tokenizer, prompts)
    else:
        raise ValueError(f"Invalid benchmark: {benchmark}")