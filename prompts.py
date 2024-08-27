import random
import numpy as np

def format_example_mmlu(data, include_answer=True):
    choices = ["A", "B", "C", "D"]
    prompt = data["question"]
    k = len(data["options"])
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], data["options"][j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(data["answer"])
    return prompt

def gen_prompt_mmlu(trainset, example_num, eval_times, subject=None):
    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s
    prompt_list = []
    random.seed(42)
    np.random.seed(42)
    for j in range(eval_times):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
        trainset = trainset.shuffle()
        for i in range(example_num):
            prompt += format_example_mmlu(trainset[i])
        prompt_list.append(prompt)
    return prompt_list


def format_example_flan(data, include_answer=True, include_question=True):
    prompt = ""
    if include_question:
        prompt += " Input:\n"
        prompt += data["inputs"]
        prompt += "\n"
    if include_answer:
        prompt += " Output:\n"
        prompt += data["targets"]
        prompt += "\n\n"
    return prompt

def gen_prompt_flan(trainset, example_num, eval_times):
    prompt_list = []
    random.seed(42)
    np.random.seed(42)
    for j in range(eval_times):    
        prompt = "You are a helpful AI assistant. Here are some example input-output pairs that you should follow.\n\n"
        trainset = trainset.shuffle()
        for i in range(example_num):
            prompt += format_example_flan(trainset[i])
        prompt_list.append(prompt)
    return prompt_list

def format_example_bbh(data):
    prompt = "Q: "
    prompt += data["input"]
    prompt += "\n"
    prompt += "A: Let's think step by step."
    return prompt

def gen_prompt_bbh(trainset, example_num, eval_times):
    return ["You are given a task with examples. You need to answer the question based on the examples. You need to think step by step and conclude with \"So the answer is <ANSWER>.\" in the end as in the examples.\n" + trainset[0]["prompt"]]

def format_example_agi(data, include_answer=True):
    prompt = "Question:\n"
    prompt += data["question"]
    prompt += "\n"
    prompt += "Choices:\n"
    for choice in data["options"]:
        prompt += f"{choice}\n"
    prompt += "Answer: \n"
    if include_answer:
        prompt += data["label"]
        prompt += "\n\n"
    return prompt

def gen_prompt_agi(trainset, example_num, eval_times):
    prompt_list = []
    random.seed(42)
    np.random.seed(42)
    for j in range(eval_times):
        prompt = "The following are multiple choice questions (with answers).\n\n"
        trainset = trainset.shuffle()
        for i in range(example_num):
            prompt += format_example_agi(trainset[i])
        prompt_list.append(prompt)
    return prompt_list

def gen_prompt(task, trainset, example_num, eval_times, subject=None):
    if example_num > len(trainset):
        example_num = len(trainset)
    if "mmlu" in task:
        return gen_prompt_mmlu(trainset, example_num, eval_times, subject)
    elif "flan" in task:
        return gen_prompt_flan(trainset, example_num, eval_times)
    elif "bbh" in task:
        return gen_prompt_bbh(trainset, example_num, eval_times)
    elif "agieval" in task:
        return gen_prompt_agi(trainset, example_num, eval_times)
    else:
        raise ValueError("Invalid task name")


def format_example(task, data):
    if "mmlu" in task:
        return format_example_mmlu(data, include_answer=False)
    elif "flan" in task:
        return format_example_flan(data, include_answer=False)
    elif "bbh" in task:
        return format_example_bbh(data)
    elif "agieval" in task:
        return format_example_agi(data, include_answer=False)
    else:
        raise ValueError("Invalid task name")