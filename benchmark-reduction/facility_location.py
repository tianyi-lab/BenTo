from submodlib import FacilityLocationFunction
import numpy as np
import csv
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.spatial import distance
from math import sqrt

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

acc_list = []
for t in range(10):
    acc_list.append(load_csv(f"log/csv/acc_{t}.csv"))
acc_array = np.mean(acc_list, axis=0)
acc_array = StandardScaler(with_std=False).fit_transform(acc_array)
acc_array = distance.squareform(distance.pdist(acc_array))
acc_max = np.max(acc_array)
acc_array = 1.5*acc_max - acc_array
acc_array = find_optimal_ordering(acc_array)

#objFL = FacilityLocationFunction(n=52, data=acc_array_others, mode="dense", separate_rep=True, n_rep=5, data_rep=acc_array_target)
objFL = FacilityLocationFunction(n=57, data=acc_array, mode="dense", separate_rep=False)
#objFL = FacilityLocationFunction(n=57, sijs=acc_array, mode="dense", separate_rep=False)
greedyList = objFL.maximize(budget=10,optimizer='LazyGreedy',show_progress=True)
mylist = [a[0] for a in greedyList]
print(mylist)
