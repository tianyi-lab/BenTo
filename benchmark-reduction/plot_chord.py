import pandas as pd
import holoviews as hv
from holoviews import opts, dim
import numpy as np
from sklearn.preprocessing import StandardScaler


import csv

#hv.extension('bokeh')
hv.extension('matplotlib')
hv.output(size=200)

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

clusters = [
    [21, 30, 31],               # Cluster 0
    [17, 22, 23, 41, 53],       # Cluster 1
    [0, 7, 11, 13, 20, 37],     # Cluster 2
    [28, 32, 33, 43, 46, 49, 50],  # Cluster 3
    [1, 4, 5, 9, 18, 44, 48],   # Cluster 4
    [34, 40, 52, 54, 55, 56],   # Cluster 5
    [8, 15, 16, 25, 29],        # Cluster 6
    [3, 14, 24, 26, 38, 39, 47, 51], # Cluster 7
    [2, 6, 10, 12, 19, 27],     # Cluster 8
    [35, 36, 42, 45]            # Cluster 9
]


new_TASKS = []
new_clusters = []
for cluster in clusters:
    new_TASKS.extend([TASKS[i] for i in cluster])
    new_clusters.append([new_TASKS.index(TASKS[i]) for i in cluster])


def load_csv(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return np.array(data[1:], dtype=float)

def rotate_label(plot, element):    
    labels = plot.handles["labels"]
    for annotation in labels:        
        angle = annotation.get_rotation()
        if 90 < angle < 270:
            annotation.set_rotation(180 + angle)
            annotation.set_horizontalalignment("right")

acc_list = []
for t in range(10):
    acc_list.append(load_csv(f"log/csv/acc_{t}.csv"))
acc_array = np.mean(acc_list, axis=0)
acc_array = StandardScaler().fit_transform(acc_array)

links = []
for i in range(57):
    for j in range(57):
        if i != j:
            links.append((new_TASKS.index(TASKS[i]), new_TASKS.index(TASKS[j]), acc_array[i, j]))

links = pd.DataFrame(links, columns=['source', 'target', 'value'])
groups = [0]*57
for i, cluster in enumerate(new_clusters):
    for j in cluster:
        groups[j] = i

hex_colors = [
    "#800080" ,  # purple
    "#33ff57",  # spring green
    "#3357ff",  # royal blue
    "#ffd700",  # golden yellow
    "#ff69b4",  # hot pink
    "#8a2be2",  # blue violet
    "#00ced1",  # dark turquoise
    "#ff4500",  # orange red
    "#7fff00",  # chartreuse
    "#d2691e"   # chocolate
]

edge_hex_colors =[]
for i, cluster in enumerate(new_clusters):
    edge_hex_colors.extend([hex_colors[i]]*len(cluster))

nodes = hv.Dataset(pd.DataFrame({'name': new_TASKS, 'group':groups}), 'index')
chord = hv.Chord((links,nodes), vdims="value").select(value=(1.5, None))
chord.opts(opts.Chord(cmap=hex_colors, edge_cmap=edge_hex_colors, edge_color=dim('source'), labels='name', node_color=dim('group'),hooks=[rotate_label]))
hv.save(chord, "images/chord.pdf", fmt='pdf')
hv.save(chord, "images/chord.png", fmt='png')