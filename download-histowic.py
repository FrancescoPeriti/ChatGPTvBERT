import os
import json
import string
import random
import numpy as np
import pandas as pd
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict

# download original data
subprocess.run("wget https://zenodo.org/record/5796878/files/dwug_en.zip?download=1", shell=True)

# unzip data
subprocess.run("unzip 'dwug_en.zip?download=1'", shell=True)

# rename
os.rename("dwug_en", "dwug_en_tmp")

# The Answer to the Great Question of Life, the Universe and Everything is Forty-two
random.seed(42)

# create dir
Path('data/HistoWiC').mkdir(parents=True, exist_ok=True)
Path('data/GradedHistoWiC').mkdir(parents=True, exist_ok=True)

# sentence pair
s1_gold = list() # first sentence list with continuous score
s2_gold = list() # second sentence list with continuous score
s1_labels = list() # first sentence list with binary score
s2_labels = list() # second sentence list with binary score

# identifier
idx=0
for f_j, f_u in zip(Path(f'dwug_en_tmp/data/').glob('**/judgments.csv'), Path(f'dwug_en_tmp/data/').glob('**/uses.csv')):
    # File judgment, File uses
    f_j, f_u = str(f_j), str(f_u)

    # Lemma
    lemma = f_j.split('/')[2]

    # part of speech
    if '_nn' in lemma:
        pos = 'N'
    if '_vb' in lemma:
        pos = 'V'
    else:
        pos = 'unknown'

    # Uses
    uses = open(f_u, mode='r', encoding='utf-8').readlines()
    columns = uses[0][:-1].split('\t')
    uses_dict = dict()
    for i, row in enumerate(uses[1:]):
        row = dict(zip(columns, row[:-1].split('\t')))
        try:
            start, end = row['indexes_target_token'].split(':')
        except:
            start = row['indexes_target_token'][0]
            end = row['context'][int(start):].strip()
            # safe check
            if end[-1] in string.punctuation:
                end = int(start) + len(end[:-1])
            else:
                end = int(start) + len(end)

        start, end = int(start), int(end)
        # new use record
        uses_dict[row['identifier']] = dict(lemma=lemma,
                                            pos=pos,
                                            token=row['context'][int(start):int(end)],
                                            start=int(start), end=int(end),
                                            sent=row['context'],
                                            grouping=row['grouping'])

    # Judgments
    judgments_df = open(f_j, mode='r', encoding='utf-8').readlines()
    columns = judgments_df[0][:-1].split('\t')

    # get number of judgemnts for each pair
    judgements_count = defaultdict(lambda: defaultdict(int))
    for i, row in enumerate(judgments_df[1:]):
        row = dict(zip(columns, row[:-1].split('\t')))

        # idx pair
        idx_sorted = sorted([row['identifier1'], row['identifier2']])

        # judgment score
        score = row['judgment']

        # store info
        judgements_count[idx_sorted[0] + ' ' + idx_sorted[1]][score] += 1

    # assign binary label according to the maximum judgment agreement
    for k in list(judgements_count.keys()):
        judgments = list(judgements_count[k].keys())
        counts = list(judgements_count[k].values())
        n_annotators = sum(counts)

        # for the sake of quality we do not rely to single evaluation
        if n_annotators < 2:
            continue

        judgment = np.array([int(eval(j)) for j in judgments]).mean()

        if 3.5 <= judgment <=4:
            label = 1
            gold = judgment
        elif 1 <=judgment <= 1.5: #1:
            label = 0
            gold = judgment
        else:
            continue
        
        identifier1, identifier2 = k.split()

        token1 = uses_dict[identifier1]['token']
        sent1 = uses_dict[identifier1]['sent']
        pos1 = uses_dict[identifier1]['pos']
        start1 = sent1.find(token1)
        end1 = start1 + len(token1)
        gold1 = gold
        label1 = label

        token2 = uses_dict[identifier2]['token']
        sent2 = uses_dict[identifier2]['sent']
        pos2 = uses_dict[identifier2]['pos']
        start2 = sent2.find(token2)
        end2 = start2 + len(token2)
        gold2=gold
        label2 = label

        # only pairs from different time periods
        if uses_dict[identifier1]['grouping'] == uses_dict[identifier2]['grouping']:
            continue

        s1_labels.append(dict(id=idx, lemma=lemma, token=token1,
                       start=start1, end=end1,
                       pos=pos1,
                       sentence=sent1, gold=label1))
        s2_labels.append(dict(id=idx, lemma=lemma, token=token2,
                       start=start2, end=end2,
                       pos=pos2,
                       sentence=sent2, gold=label2))
        s1_gold.append(dict(id=idx, lemma=lemma, token=token1,
                       start=start1, end=end1,
                       pos=pos1,
                       sentence=sent1, gold=gold1))
        s2_gold.append(dict(id=idx, lemma=lemma, token=token2,
                       start=start2, end=end2,
                       pos=pos2,
                       sentence=sent2, gold=gold2))
        # new id 
        idx+=1

idx = list(range(0, len(s1_labels)))
random.shuffle(idx)
s1_labels = np.array(s1_labels)[idx]
s2_labels = np.array(s2_labels)[idx]

s1_gold = np.array(s1_gold)[idx]
s2_gold = np.array(s2_gold)[idx]

percentage = 0.6
n_train = int(len(s1_labels) * percentage)

percentage = 0.01
n_trial = int(len(s1_labels) * percentage)
n_test = len(s1_labels) - n_trial - n_train

print(f'# Train: {n_train}, # Test: {n_test}, # Trial: {n_trial}')

## Train set
# wrapper for processed data
data_labels, data_gold = list(), list()
for i in range(0, n_train):
    data_labels.append(json.dumps(s1_labels[i])+'\n')
    data_labels.append(json.dumps(s2_labels[i])+'\n')
    data_gold.append(json.dumps(s1_gold[i])+'\n')
    data_gold.append(json.dumps(s2_gold[i])+'\n')

# store data
with open('data/HistoWiC/train.txt', mode='w', encoding='utf-8') as f:
    f.writelines(data_labels)
with open('data/GradedHistoWiC/train.txt', mode='w', encoding='utf-8') as f:
    f.writelines(data_gold)

## Test set
# wrapper for processed data
data_labels, data_gold = list(), list()
for i in range(n_train, n_train+n_test):
    data_labels.append(json.dumps(s1_labels[i])+'\n')
    data_labels.append(json.dumps(s2_labels[i])+'\n')
    data_gold.append(json.dumps(s1_gold[i])+'\n')
    data_gold.append(json.dumps(s2_gold[i])+'\n')

# store data
with open('data/HistoWiC/test.txt', mode='w', encoding='utf-8') as f:
    f.writelines(data_labels)
with open('data/GradedHistoWiC/test.txt', mode='w', encoding='utf-8') as f:
    f.writelines(data_gold)

## Trial set
# wrapper for processed data
data_labels, data_gold = list(), list()
for i in range(n_train+n_test, len(s1_labels)):
    data_labels.append(json.dumps(s1_labels[i])+'\n')
    data_labels.append(json.dumps(s2_labels[i])+'\n')
    data_gold.append(json.dumps(s1_gold[i])+'\n')
    data_gold.append(json.dumps(s2_gold[i])+'\n')

# store data
with open('data/HistoWiC/trial.txt', mode='w', encoding='utf-8') as f:
    f.writelines(data_labels)
with open('data/GradedHistoWiC/trial.txt', mode='w', encoding='utf-8') as f:
    f.writelines(data_gold)

# remove zenodo directory
shutil.rmtree("dwug_en_tmp")
os.remove('dwug_en.zip?download=1')
