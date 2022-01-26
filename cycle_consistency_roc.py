import json
from sklearn.metrics import roc_curve, auc 

with open('./train_entries_list.json') as f:
    spider = json.load(f)
    spider = [{'exec': 1 if d['exec'] else 0, 'partial': sum([d['partial'][k]['f1'] for k in d['partial']])} for d in spider]

with open('./train_negative_entries_list.json') as f:
    negative = json.load(f)
    negative = [{'exec': 1 if d['exec'] else 0, 'partial': sum([d['partial'][k]['f1'] for k in d['partial']])} for d in negative]

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
    print(auc(fpr, tpr))
