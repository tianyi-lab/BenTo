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


from collections import Counter
import csv

icl_results = {}
acc = [np.zeros((57,57)) for _ in range(10)]
for i, task in enumerate(TASKS):
    json_name = f"log/smallrange/normal/raw_results/run_results_llama_13b_{task}_10.json"
    with open(json_name, 'r') as f:
        icl_results[i] = json.load(f)

    for j, task2 in enumerate(TASKS):
        pred_answers_list_icl = [[icl_results[i][task2]["pred_answers"][k] for k in range(len(icl_results[i][task2]["pred_answers"])) if k % 10 == j] for j in range(10)]
        gold_answers = [icl_results[i][task2]["gold_answers"][k] for k in range(len(icl_results[i][task2]["gold_answers"])) if k % 10 == 0]
        
        for t in range(10):
            n = len(gold_answers)
            for pred, gold in zip(pred_answers_list_icl[t], gold_answers):
                if pred == gold: acc[t][i][j] += 1
            acc[t][i][j] /= n

#save acc to 10 csv files
for t in range(10):
    with open(f"log/csv/acc_{t}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(TASKS)
        for i in range(57):
            writer.writerow(acc[t][i])