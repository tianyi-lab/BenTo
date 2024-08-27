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
from math import sqrt
import random

from rank_bm25 import BM25Okapi

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


acc_array = np.zeros((len(TASKS), len(TASKS)))
all_corpse = []
for task in TASKS:
    with open(f'data/test/{task}_test.csv', "r") as f:
        reader = csv.reader(f)
        task_data = list(reader)
        flat_corpse = []
        for row in task_data:
            flat_corpse.extend(row)
        tokenized_corpse = [doc.split(" ") for doc in flat_corpse]
        flat_tokenized_corpse = []
        for doc in tokenized_corpse:
            flat_tokenized_corpse.extend(doc)
        all_corpse.append(flat_tokenized_corpse)

# bm25 = BM25Okapi(all_corpse)
# for i, doc in enumerate(all_corpse):
#     doc_scores = bm25.get_scores(doc)
#     for j, score in enumerate(doc_scores):
#         acc_array[i, j] = score


acc_list = []
for t in range(10):
    acc_list.append(load_csv(f"log/csv/acc_{t}.csv"))
acc_array = np.mean(acc_list, axis=0)
#var_acc = np.var(acc_list, axis=0)
#acc_array = np.concatenate((acc_array, var_acc), axis=1)
acc_array = StandardScaler(with_std=False).fit_transform(acc_array)
acc_max = np.max(distance.squareform(distance.pdist(acc_array)))
random.seed(42)
test_list = []

c = 1.5
acc_array_new = c*acc_max - distance.squareform(distance.pdist(acc_array))
#acc_array_new = find_optimal_ordering(acc_array_new)
objFL = FacilityLocationFunction(n=57, sijs=acc_array_new, mode="dense", separate_rep=False)
greedyList = objFL.maximize(budget=10,optimizer='LazyGreedy')
greedyList = [t[0] for t in greedyList]
test_list.append(greedyList)

for _ in range(1000):
    c = random.random()*49+1
    acc_array_new = c*acc_max - distance.squareform(distance.pdist(acc_array))
    #acc_array_new = find_optimal_ordering(acc_array_new)
    objFL = FacilityLocationFunction(n=57, sijs=acc_array_new, mode="dense", separate_rep=False)
    greedyList = objFL.maximize(budget=10,optimizer='LazyGreedy')
    greedyList = [t[0] for t in greedyList]
    #greedyList = random.choices(range(57), k=10)
    test_list.append(greedyList)



def extract_acc_data(file_path):
    acc_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ACC-"):
                parts = line.split()
                acc_value = float(parts[1]) if len(parts) > 1 else None
                acc_data.append(acc_value)
    return acc_data


path = "log/benchmark"
test_list = [[t[:i] for i in range(1, 11)] for t in test_list]
all_indices_list = test_list
length = []
with open("length.out") as file:
    for line in file:
        length.append(int(line))

def compute_acc(all_acc, indices, all_length):
    sum_length = sum([all_length[i] for i in indices])
    acc = sum([all_acc[i]*all_length[i]/sum_length for i in indices])
    return acc

label_list = ["ground_truth"] + [f"test_{i}" for i in range(1001)]
error_list = [[] for _ in range(1001)]

for p in range(10):
    indices_list = [k[p] for k in all_indices_list]
    #print(f"Using {p+1} tasks")
    acc_dict = {}
    #label_list = ["ground_truth", "random1", "random2", "random3", "random4", "random5", "chatgpt", "facilityloc", "facilityloc_spec", "facilityloc_spec_dist"]
    for file_name in os.listdir(path):
        if file_name.endswith(".out") and 't5' not in file_name and "gpt2" not in file_name:
            file_path = os.path.join(path, file_name)
            acc_data = extract_acc_data(file_path)
            gt_performance = acc_data[-1]
            acc_data = acc_data[:-1]
            acc_dict[file_name] = []
            acc_dict[file_name].append(gt_performance)
            for indices in indices_list:
                acc = compute_acc(acc_data, indices, length)
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
    if sum(e) < sum(error_list[0]):
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