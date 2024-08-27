import os
import matplotlib.pyplot as plt
import math

from submodlib import FacilityLocationFunction
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.spatial import distance
from math import sqrt, exp
import random

from rank_bm25 import BM25Okapi

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

def load_csv(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return np.array(data[1:], dtype=float)

def separate_rows(arr, indices):
    # Ensure indices are a numpy array for advanced indexing
    indices = np.array(indices)
    
    # Get all row indices of the array
    all_indices = np.arange(arr.shape[0])
    
    # Find the indices that are not in the specified list
    mask = np.isin(all_indices, indices, invert=True)
    
    # Rows with specified indices
    rows_with_indices = arr[indices]
    
    # Rows without the specified indices
    rows_without_indices = arr[mask]
    
    return rows_with_indices, rows_without_indices

def find_optimal_ordering(matrix):
    # Step 1: Assuming matrix is similarity; otherwise compute similarity matrix
    similarity_matrix = matrix

    # Step 2: Compute the graph Laplacian
    graph_laplacian = laplacian(similarity_matrix, normed=True)

    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = eigh(graph_laplacian)

    return eigenvectors

random.seed(42)
test_list = []
acc_array = load_csv(f"log/csv/neg_loss_flan.csv")
acc_array = np.exp(acc_array)
#acc_array = np.load("log/bm25/bm25_score_matrix_FLAN.npy") 
#acc_array = acc_array / (np.sum(acc_array, axis=0)+1e-12)
acc_array = StandardScaler(with_std=True).fit_transform(acc_array)
acc_max = np.max(distance.squareform(distance.pdist(acc_array)))
acc_array = 1.5*acc_max - distance.squareform(distance.pdist(acc_array))
#acc_array = find_optimal_ordering(acc_array)
objFL = FacilityLocationFunction(n=66, sijs=acc_array, mode="dense", separate_rep=False)
greedyList = objFL.maximize(budget=12,optimizer='LazyGreedy',show_progress=True)
greedyList = [a[0] for a in greedyList]
test_list.append(greedyList)

for _ in range(1000):
    c = random.random()*49+1
    #greedyList = random.choices(range(66), k=12)
    acc_array = load_csv(f"log/csv/neg_loss_flan.csv")
    acc_array = np.exp(acc_array)
    #acc_array = np.load("log/bm25/bm25_score_matrix_FLAN.npy") 
    #acc_array = acc_array / (np.sum(acc_array, axis=0)+1e-12)
    acc_array = StandardScaler(with_std=True).fit_transform(acc_array)
    acc_max = np.max(distance.squareform(distance.pdist(acc_array)))
    acc_array = c*acc_max - distance.squareform(distance.pdist(acc_array))
    #acc_array = find_optimal_ordering(acc_array)
    objFL = FacilityLocationFunction(n=66, sijs=acc_array, mode="dense", separate_rep=False)
    greedyList = objFL.maximize(budget=12,optimizer='LazyGreedy',show_progress=True)
    greedyList = [a[0] for a in greedyList]
    test_list.append(greedyList)


def extract_acc_data(file_path):
    acc_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Eval"):
                parts = line.split()
                acc_value = float(parts[-1])
                acc_data.append(acc_value)
    return acc_data


path = "log/benchmark_flan"
test_list = [[t[:i] for i in range(1, 13)] for t in test_list]
all_indices_list = test_list

def compute_acc(all_acc, indices):
    acc = sum([all_acc[i] for i in indices])/len(indices)
    return acc

label_list = ["ground_truth"] + [f"test_{i}" for i in range(1001)]
error_list = [[] for _ in range(1001)]

for p in range(12):
    indices_list = [k[p] for k in all_indices_list]
    print(f"Using {p+1} tasks")
    acc_dict = {}
    #label_list = ["ground_truth", "random1", "random2", "random3", "random4", "random5", "chatgpt", "facilityloc", "facilityloc_spec", "facilityloc_spec_dist"]
    for file_name in os.listdir(path):
        if file_name.endswith(".out"):
            file_path = os.path.join(path, file_name)
            acc_data = extract_acc_data(file_path)
            gt_performance = compute_acc(acc_data, list(range(66)))
            acc_dict[file_name] = []
            acc_dict[file_name].append(gt_performance)
            for indices in indices_list:
                acc = compute_acc(acc_data, indices)
                acc_dict[file_name].append(acc)

    # #plot one line for each label, x_axis is file_name
    # plt.figure(figsize=(10, 10))
    # for i in range(1, len(label_list)):
    #     plt.plot([acc_dict[file_name][i] for file_name in acc_dict], label=label_list[i])
    # #the first line should be more thick
    # plt.plot([acc_dict[file_name][0] for file_name in acc_dict], label=label_list[0], linewidth=3)
    # #the second to the sixth line should have the same color
    # plt.gca().get_lines()[0].set_color('black')
    # plt.gca().get_lines()[1].set_color('black')
    # plt.gca().get_lines()[2].set_color('black')
    # plt.gca().get_lines()[3].set_color('black')
    # plt.gca().get_lines()[4].set_color('black')
    # plt.gca().get_lines()[5].set_color('red')
    # plt.gca().get_lines()[6].set_color('blue')
    # plt.gca().get_lines()[7].set_color("aquamarine")
    # #change the x-axis to file_name
    # plt.xticks(range(len(acc_dict)), [k[:-4] for k in acc_dict.keys()])


    # plt.legend()
    # plt.savefig(f"images/benchmark_{p+1}tasks.png")

    #compute error for each method compared to ground truth
    all_gt_result = [acc_dict[file_name][0] for file_name in acc_dict]
    for i in range(1, len(label_list)):
        #error = [acc_dict[file_name][i] for j, file_name in enumerate(acc_dict)]
        error = [acc_dict[file_name][i] - all_gt_result[j] for j, file_name in enumerate(acc_dict)]
        error = math.sqrt(sum([e*e for e in error])/len(error))/math.sqrt(sum([e*e for e in all_gt_result])/len(all_gt_result))
        #print(label_list[i], error)
        error_list[i-1].append(error)

count = 0
for e in error_list[1:]:
    if sum(e) <= sum(error_list[0]):
        count += 1
print(count)

# error_data = np.array(error_list)
# mean_error = np.mean(error_data, axis=0)
# one_std_above_mean_error = mean_error + np.std(error_data, axis=0)
# one_std_below_mean_error = mean_error - np.std(error_data, axis=0)
# print(mean_error.tolist())
# print(one_std_above_mean_error.tolist())
# print(one_std_below_mean_error.tolist())

    # #compute correlation of each method with the ground truth
    # import numpy as np
    # all_gt_result = np.array(all_gt_result)
    # for i in range(1, len(label_list)):
    #     all_result = np.array([acc_dict[file_name][i] for file_name in acc_dict])
    #     correlation = np.corrcoef(all_gt_result, all_result)[0, 1]
    #     print(label_list[i], correlation)
        
    # #compute spearmann correlation of each method with the ground truth
    # from scipy import stats
    # for i in range(1, len(label_list)):
    #     correlation = stats.spearmanr(all_gt_result, all_result)
    #     print(label_list[i], correlation)

    # #compute kendall correlation of each method with the ground truth
    # for i in range(1, len(label_list)):
    #     correlation = stats.kendalltau(all_gt_result, all_result)
    #     print(label_list[i], correlation)

# print("Mean results")
# mean_result_list = []
# for i in range(1, len(label_list)):
#     for j, file_name in enumerate(acc_dict):
#         mean_result_list.append(sum([error_list[i-1][p][j] for p in ])/10)