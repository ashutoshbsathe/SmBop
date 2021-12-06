import smbop.utils.ra_preproc as ra_preproc
from smbop.utils import moz_sql_parser as msp 
from nearest_trees import CustomConfig 
import json 
import pickle 
from tqdm import tqdm 
from apted import APTED
import numpy as np 
import os 
from anytree import PreOrderIter
import itertools 
import multiprocessing 

TRAIN_PATH = '../rat-sql/data/spider/train_spider.json'

indices = []
num_skipped = 0

with open(TRAIN_PATH) as f:
    train = json.load(f)

if not os.path.exists('./valid_samples.pkl') or not os.path.exists('./valid_indices.pkl'):
    samples = []
    print('Checking if all samples can be parsed')
    for i, sample in enumerate(tqdm(train)):
        try:
            tree_dict = msp.parse(sample['query'])
            tree_obj = ra_preproc.ast_to_ra(tree_dict['query'])
            size = len(list(PreOrderIter(tree_obj)))
            samples.append((tree_dict, tree_obj, size))
            indices.append(i)
        except Exception as e:
            num_skipped += 1
            samples.append((None, None, None))
            #print(f'Skipped index {i}, num_skipped={num_skipped}, {type(e)}')
            continue
    print(f'Skipped {num_skipped} samples in total')
    with open('valid_samples.pkl', 'wb') as f:
        pickle.dump(samples, f)
    with open('valid_indices.pkl', 'wb') as f:
        pickle.dump(indices, f)
else:
    with open('valid_samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    with open('valid_indices.pkl', 'rb') as f:
        indices = pickle.load(f)

dist_matrix = np.ones((len(train), len(train))) * 1e9
size_array = np.zeros(len(train))
config = CustomConfig()

idxs = list(itertools.combinations_with_replacement(range(len(indices)), 2))

def process_i_j(idx):
    i, j = idx 
    _, tree_obj_i, size_i = samples[i]
    if tree_obj_i is None:
        size_array[i] = 1e-6 
        return 
    else:
        size_array[i] = size_i
    _, tree_obj_j, size_j = samples[j]
    if tree_obj_j is None:
        return 
    dist = APTED(tree_obj_i, tree_obj_j, config).compute_edit_distance()
    assert dist >= 0
    normalized = dist / max(size_j, size_i)
    return normalized

with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
    res = list(tqdm(p.imap(process_i_j, idxs), total=len(idxs)))

for res_i, idx in enumerate(idxs):
    i, j = idx
    normalized = res[res_i]
    dist_matrix[j, i] = normalized
    dist_matrix[i, j] = normalized

with open('neighbour_data.pkl', 'wb') as f:
    pickle.dump({
        'dists': dist_matrix,
        'sizes': size_array,
        'indices': indices,
    }, f)
