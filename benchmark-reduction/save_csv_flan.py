import os
import numpy as np
import csv

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

acc = np.zeros((66,66))

def extract_acc_data(file_path):
    acc_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Eval"):
                parts = line.split()
                acc_value = -float(parts[-1])
                acc_data.append(acc_value)
    return acc_data

log_path = "log/flan_sp"
path_list = os.listdir(log_path)
to_sort = [int(p.split('=')[-1].split('.')[0]) for p in path_list]
path_list = [x for _, x in sorted(zip(to_sort, path_list))]
for i, path in enumerate(path_list):
    acc_data = extract_acc_data(f"{log_path}/{path}")
    acc[i] = acc_data

#save acc to 10 csv files
with open(f"log/csv/neg_loss_flan.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(TASKS)
    for i in range(66):
        writer.writerow(acc[i])