import json
import pandas as pd
from pathlib import Path

# templates
task = "Task: Determine whether two given sentences use a target word with the same meaning or different meanings in their respective contexts."
info = """Sentences: \nSentence 1: {}\nSentence 2: {}\n\nTarget: {}\n\nQuestion: Do the target word in both sentences have the same meaning in their respective contexts? Choose one: "True" if the target word has the same meaning in both sentences, "False" if the target word has different meanings in the sentences\n\nAnswer:"""
training = "I'll provide some examples before testing you."

# datasets
for folder in ['TempoWiC', 'HistoWiC']:
    # load train, test, trial set
    train = pd.read_json(f'data/{folder}/train.txt', lines=True)
    test = pd.read_json(f'data/{folder}/test.txt', lines=True)
    trial = pd.read_json(f'data/{folder}/trial.txt', lines=True)

    
    # few-shot prompting
    fsp = list([training + "\n"])
    for i in range(0, trial.shape[0], 2):
        sentence1 = trial.loc[i]['sentence']
        sentence2 = trial.loc[i+1]['sentence']
        target = trial.loc[i]['lemma']
        gold = trial.loc[i]['gold']

        # first occurrence introduce examples
        if i == 0:
            tmp = 'Examples:\n\n'
        else:
            tmp = ''
        
        record = tmp + task + '\n\n' + info.format(sentence1, sentence2, target) + f' {str(bool(gold))}'
        fsp.append(record.replace('\n', '##newline##') + '\n')
    fsp.append('Your turn:'+'\n')

    # trained prompting
    tp = list([training + "\n"])
    for i in range(0, trial.shape[0], 2):
        sentence1 = train.loc[i]['sentence']
        sentence2 = train.loc[i+1]['sentence']
        target = train.loc[i]['lemma']
        gold = train.loc[i]['gold']

        # first occurrence introduce examples
        if i == 0:
            tmp = 'Examples:\n\n'
        else:
            tmp = ''
            
        record = tmp + task + '\n\n' + info.format(sentence1, sentence2, target) + f' {str(bool(gold))}'
        tp.append(record.replace('\n', '##newline##') + '\n')
    tp.append('Your turn:'+'\n')    
    
    # zero-shot prompting
    zsp = list()
    for i in range(0, test.shape[0], 2):
        sentence1 = test.loc[i]['sentence']
        sentence2 = test.loc[i+1]['sentence']
        target = test.loc[i]['lemma']
        record = task + '\n\n' + info.format(sentence1, sentence2, target)
        zsp.append(record.replace('\n', '##newline##') + '\n')
    fsp.extend(zsp)
    tp.extend(zsp)

    Path(f'prompt-data/{folder}').mkdir(exist_ok=True, parents=True)
    
    # store prompts
    for p in ['zsp', 'fsp', 'tp']:
        with open(f'prompt-data/{folder}/{p}.txt', mode='w', encoding='utf-8') as f:
            f.writelines(eval(p))
