import json
import pickle 

with open('./our_entries_list.json') as f:
    our = json.load(f)

with open('./orig_entries_list.json') as f:
    orig = json.load(f)

with open('../ms_rnd1/finetune_pipeline/expt_modular/world_1/new_split_frac=0.667;prop_n=10,k=4,tree=1e-3,thresh=20,filter=1;sql2nl_n=4,k=1,filter=0;/dev.pkl', 'rb') as f:
    data = pickle.load(f)

assert len(our) == len(orig)

for i in range(len(our)):
    assert our[i]['hardness'] == orig[i]['hardness']
    if our[i]['exact'] == False and orig[i]['exact'] == True:
        print('index = ', i)
        print('hardness = ', our[i]['hardness'])
        print('predicted SQL by finedtuned model = ', our[i]['predictSQL'])
        print('predicted SQL by original model = ', orig[i]['predictSQL'])
        print('gold SQL = ', orig[i]['goldSQL'])
        print('question = ', data[i])

