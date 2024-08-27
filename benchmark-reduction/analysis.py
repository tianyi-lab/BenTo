import os
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
import csv
import random

CLUSTER_NUM = 10


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

def find_optimal_ordering(matrix, num_clusters, matrix_to_rearrange):
    # Step 1: Assuming matrix is similarity; otherwise compute similarity matrix
    similarity_matrix = matrix

    # Step 2: Compute the graph Laplacian
    graph_laplacian = laplacian(similarity_matrix, normed=True)

    # Step 3: Eigenvalue decomposition
    eigenvalues, eigenvectors = eigh(graph_laplacian, subset_by_index=[1, num_clusters])

    # Step 4: Clustering of Eigenvectors (using k-means here)
    kmeans = KMeans(n_clusters=num_clusters, n_init="auto", random_state=0).fit(eigenvectors)
    labels = kmeans.labels_

    # Step 5: Rearrange the matrix according to the labels
    idx = np.argsort(labels)
    rearranged_matrix = matrix_to_rearrange[idx, :][:, idx]

    return rearranged_matrix, labels, idx

def load_csv(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return np.array(data[1:], dtype=float)

acc_list = []
for t in range(10):
    acc_list.append(load_csv(f"log/csv/acc_{t}.csv"))
acc_array = np.mean(acc_list, axis=0)
var_acc = np.var(acc_list, axis=0)

normalized_acc = StandardScaler(with_std=False).fit_transform(acc_array)


random.seed(42)
max_acc = np.max(distance.squareform(distance.pdist(normalized_acc)))
final_acc = 1.5*max_acc - distance.squareform(distance.pdist(normalized_acc))
reordered_matrix, labels, _ = find_optimal_ordering(final_acc, CLUSTER_NUM, final_acc)
plt.figure(figsize=(10, 8))
plt.imshow(reordered_matrix, cmap='plasma')
plt.colorbar()
#plt.title('Heatmap of Reordered Matrix')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig(f"images/heatmap_{CLUSTER_NUM}_final.png")
plt.savefig(f"images/heatmap_{CLUSTER_NUM}_final.pdf")

for i in np.unique(labels):
    print(f"The following tasks are in cluster {i}")
    for j in np.where(labels == i)[0]:
        print(TASKS[j])
    print("\n")


