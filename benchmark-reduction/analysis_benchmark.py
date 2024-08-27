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
            if line.startswith("ACC-"):
                parts = line.split()
                acc_value = float(parts[1]) if len(parts) > 1 else None
                acc_data.append(acc_value)
    return acc_data


path = "log/benchmark"
chatgpt = [7, 9, 31, 3, 14, 37, 48, 15, 2, 53]
chatgpt = [chatgpt[:i] for i in range(1, 11)]
bm25 = [50, 43, 7, 47, 3, 12, 17, 15, 55, 44]
bm25 = [bm25[:i] for i in range(1, 11)]
bm25_sim = [53, 48, 21, 54, 29, 50, 43, 31, 12, 30]
bm25_sim = [bm25_sim[:i] for i in range(1, 11)]
facilityloc = [24, 0, 45, 10, 3, 2, 8, 48, 4, 50]
facilityloc = [facilityloc[:i] for i in range(1, 11)]
temp = [51, 9, 25, 35, 20, 22, 27, 30, 33, 24]
temp = [temp[:i] for i in range(1, 11)]

all_indices_list = [bm25,bm25_sim, facilityloc, temp]
length = []
with open("length.out") as file:
    for line in file:
        length.append(int(line))

def compute_acc(all_acc, indices, all_length):
    sum_length = sum([all_length[i] for i in indices])
    acc = sum([all_acc[i]*all_length[i]/sum_length for i in indices])
    return acc

label_list = ["ground_truth","bm25-le","bm25-sim", "ours-le", "ours-sim"]
error_list = [[], [],[], []]

for p in range(10):
    indices_list = [k[p] for k in all_indices_list]
    print(f"Using {p+1} tasks")
    acc_dict = {}
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

    #compute error for each method compared to ground truth
    all_gt_result = [acc_dict[file_name][0] for file_name in acc_dict]
    print(all_gt_result)
    for i in range(1, len(label_list)):
        #error = [acc_dict[file_name][i] for j, file_name in enumerate(acc_dict)]
        error = [abs(acc_dict[file_name][i]-all_gt_result[j]) for j, file_name in enumerate(acc_dict)]
        error = math.sqrt(sum([e*e for e in error])/len(error))/math.sqrt(sum([e*e for e in all_gt_result])/len(all_gt_result))
        print(label_list[i], error)
        error_list[i-1].append(error)

colors = ['b', 'r', 'tab:blue', 'tab:red']
plt.figure(figsize=(10, 8))
for i in range(len(label_list)-1):
    plt.plot(range(1, 11), error_list[i], label=label_list[i+1],color=colors[i])
plt.legend(fontsize=18)
plt.xlabel("Number of tasks", fontsize=20)
plt.ylabel("Normalized RMSE", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("images/benchmark.png")
plt.savefig("images/benchmark.pdf")
