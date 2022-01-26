import json
from sklearn.metrics import roc_curve, auc 
import numpy as np 

with open('./train_entries_list.json') as f:
    spider = json.load(f)
    spider = [{'exec': 1 if 'exec' in d and d['exec'] else 0, 'partial': sum([d['partial'][k]['f1'] for k in d['partial']]) if 'partial' in d else 0} for d in spider]

with open('./train_negative_entries_list.json') as f:
    negative = json.load(f)
    negative = [{'exec': 1 if 'exec' in d and d['exec'] else 0, 'partial': sum([d['partial'][k]['f1'] for k in d['partial']]) if 'partial' in d else 0} for d in negative]

with open('./dev_entries_list.json') as f:
    dev = json.load(f)
    dev = [{'exec': 1 if 'exec' in d and d['exec'] else 0, 'partial': sum([d['partial'][k]['f1'] for k in d['partial']]) if 'partial' in d else 0} for d in dev]

for exec_multiplier in [1, 2, 4, 8, 16]:
    y = []
    scores = []
    for x in spider:
        y.append(1)
        scores.append(exec_multiplier * x['exec'] + x['partial'])
    for x in negative:
        y.append(0)
        scores.append(exec_multiplier * x['exec'] + x['partial'])

    fpr, tpr, threshold = roc_curve(y, scores)
    print('AUC', auc(fpr, tpr))
    optimal_threshold = threshold[np.argmax(tpr - fpr)]
    print('Optimal threshold', optimal_threshold)

    correct = 0
    for x in dev:
        if exec_multiplier * x['exec'] + x['partial'] >= optimal_threshold:
            correct += 1

    print('Dev set accuracy', correct / len(dev))
    print(64*'-')
