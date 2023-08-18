import os
import shutil
import subprocess
import pandas as pd
from pathlib import Path


subprocess.run("git clone https://github.com/cardiffnlp/TempoWiC.git", shell=True)

# rename test filename
os.rename('TempoWiC/data/test.gold.tsv', 'TempoWiC/data/test.labels.tsv')
os.rename('TempoWiC/data/trial.gold.tsv', 'TempoWiC/data/trial.labels.tsv')
os.rename('TempoWiC/data/test-codalab-10k.data.jl', 'TempoWiC/data/test.data.jl')

# create folder
Path(f'data/TempoWiC/').mkdir(parents=True, exist_ok=True)

sets = ['train', 'test', 'validation', 'trial']

for folder in sets:
    output_filename = f'data/TempoWiC/{folder}.txt'
    
    # load labels and data
    labels = pd.read_csv(f'TempoWiC/data/{folder}.labels.tsv', sep='\t', names=['id', 'gold'])
    data = pd.read_json(f'TempoWiC/data/{folder}.data.jl', lines=True)
    
    # join dataframes
    data = data.merge(labels, on='id')
    data = data.sample(n=min(200, data.shape[0]), random_state=42)    
    
    if folder == 'validation':
        folder = 'dev'
    
    records = list()
    
    idx=0
    for _, row in data.iterrows():
        lemma = row['word']
        gold = row['gold']
        
        for sent in ['tweet1', 'tweet2']:
            sentence = row[sent]['text']
            start, end = row[sent]['text_start'], row[sent]['text_end']
            token = sentence[start:end]
            
            # avoid issues in extracting bert-embeddings
            sentence=sentence[:start] + " " + sentence[start:end] + " " + sentence[end:] # add pad space to safe occurrences
            
            # a space was added
            start+=1
            end+=1
            
            # create record
            record = dict(id=idx,
                          pos='unknwon',
                          sentence=sentence,
                          lemma=lemma,
                          token=token,
                          start=start,
                          end=end,
                          gold=gold)
            
            # add record
            records.append(record)
            
            # new id
            idx+=1
    
    # store data
    pd.DataFrame(records).to_json(output_filename, orient='records', lines=True)


# remove github clone
shutil.rmtree('TempoWiC')
