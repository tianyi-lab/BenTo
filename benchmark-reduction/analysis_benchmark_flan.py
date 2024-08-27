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
# random1 = [8, 36, 54, 51, 48, 4, 16, 7, 31, 28, 60, 61]
# random1 = [random1[:i] for i in range(1, 13)]
# random2 = [55, 54, 3, 5, 23, 23, 10, 47, 42, 19, 15, 2]
# random2 = [random2[:i] for i in range(1, 13)]
# random3 = [15, 37, 34, 8, 23, 38, 30, 40, 55, 4, 20, 10]
# random3 = [random3[:i] for i in range(1, 13)]
# random4 = [15, 19, 6, 46, 25, 30, 9, 5, 4, 1, 55, 2]
# random4 = [random4[:i] for i in range(1, 13)]
# random5 = [50, 36, 52, 5, 31, 48, 16, 2, 0, 9, 22, 58]
# random5 = [random5[:i] for i in range(1, 13)]
chatgpt = [44, 26, 46, 5, 10, 1, 24, 22, 11, 30, 43, 17]
chatgpt = [chatgpt[:i] for i in range(1, 13)]
bm25_sim = [45, 24, 25, 11, 65, 8, 46, 33, 35, 31, 14, 55]
bm25_sim = [bm25_sim[:i] for i in range(1, 13)]
bm25_lap = [46, 60, 36, 28, 34, 42, 44, 20, 26, 53, 38, 29]
bm25_lap = [bm25_lap[:i] for i in range(1, 13)]
ours_sim = [21, 34, 32, 37, 57, 11, 39, 5, 62, 2, 54, 13]
ours_sim = [ours_sim[:i] for i in range(1, 13)]
ours_lap = [55, 7, 54, 11, 44, 12, 5, 48, 0, 43, 34, 50]
ours_lap = [ours_lap[:i] for i in range(1, 13)]

all_indices_list = [chatgpt, bm25_lap, bm25_sim, ours_lap, ours_sim] #+ test_list
#all_indices_list = [bm25_lap, bm25_sim, ours_lap, ours_sim]
def compute_acc(all_acc, indices):
    acc = sum([exp(all_acc[i]) for i in indices])/len(indices)
    return acc

label_list = ["ground_truth", "chatgpt", "bm25_lap","bm25_sim", "ours_lap","ours_sim"]
error_list = [[], [], [], [], []] 
#label_list = ["ground_truth", "bm25-le","bm25-sim", "ours-le","ours-sim"]
#error_list = [[], [], [], []] 

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
        error = [abs(acc_dict[file_name][i]-all_gt_result[j]) for j, file_name in enumerate(acc_dict)]
        error = math.sqrt(sum([e*e for e in error])/len(error))/math.sqrt(sum([e*e for e in all_gt_result])/len(all_gt_result))
        print(label_list[i], error)
        error_list[i-1].append(error)

# label_list.append("random")
# error_list.append([1.2674967900402725, 1.056555025640717, 0.9185304460558078, 0.8257401243523484, 0.7667488395735004, 0.700920235577266, 0.6443405509505594, 0.6004253316999925, 0.5576765625753091, 0.5315573734880737, 0.50798628887381, 0.4894298447894774])

# print(error_list)
# print(label_list)
# exit()
colors = ['b', 'r', 'tab:blue', 'tab:red']
plt.figure(figsize=(10, 8))
for i in range(len(label_list)-1):
    plt.plot(range(1, 13), error_list[i], label=label_list[i+1],color=colors[i])
plt.legend(fontsize=18)
plt.xlabel("Number of tasks", fontsize=20)
plt.ylabel("Normalized RMSE", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("images/benchmark_flan.png")
plt.savefig("images/benchmark_flan.pdf")

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
#print(error_list)
# print("Mean results")
# for i in range(1, len(label_list)):
#     print(label_list[i], sum(error_list[i-1])/12)
# print("Best results")
# for i in range(1, len(label_list)):
#     print(label_list[i], min(error_list[i-1]))
# print("Best results position")
# for i in range(1, len(label_list)):
#     print(label_list[i], error_list[i-1].index(min(error_list[i-1])) + 1)

# print(count)
# print(count2)
