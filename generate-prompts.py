import pandas as pd
from pathlib import Path

# templates
task = "Task: Determine whether two given sentences use a target word with the same meaning or different meanings in their respective contexts."

answer = """When it's your turn, choose one: "True" if the target word has the same meaning in both sentences; "False" if the target word has different meanings in the sentences. I'll notify you when it's your turn."""

info = """Sentence 1: {}\nSentence 2: {}\nTarget: {}\nQuestion: Do the target word in both sentences have the same meaning in their respective contexts?\nAnswer: """

training = task + '\n' + """I'll provide some negative and positive examples to teach you how to deal with the task before testing you. Please respond with only "OK" during the examples; when it's your turn, answer only with "True" or "False" without any additioal text. """ + answer

no_training = task + '\n' + """Please answer only with "True" or "False" without any additional text. """ + answer 

# datasets
for folder in ['TempoWiC', 'HistoWiC']:
    # load train, test, trial set
    train = pd.read_json(f'data/{folder}/train.txt', lines=True)
    test = pd.read_json(f'data/{folder}/test.txt', lines=True)
    trial = pd.read_json(f'data/{folder}/trial.txt', lines=True)

    
    # few-shot prompting
    fsp = list([training.replace('\n', '##newline##') + "\n"])
    for i in range(0, trial.shape[0], 2):
        sentence1 = trial.loc[i]['sentence']
        sentence2 = trial.loc[i+1]['sentence']
        target = trial.loc[i]['lemma']
        gold = trial.loc[i]['gold']
        guidelines = 'This is an example. You have to answer "OK":\n'
        record = guidelines + info.format(sentence1, sentence2, target) + f' {str(bool(gold))}'
        fsp.append(record.replace('\n', '##newline##') + '\n')

    # trained prompting
    tp = list([training.replace('\n', '##newline##') + "\n"])
    truth_train = list()
    for i in range(0, train.shape[0], 2):
        sentence1 = train.loc[i]['sentence']
        sentence2 = train.loc[i+1]['sentence']
        target = train.loc[i]['lemma']
        gold = train.loc[i]['gold']
        guidelines = 'This is an example. You have to answer "OK":\n'
        record = guidelines + info.format(sentence1, sentence2, target) + f' {str(bool(gold))}'
        tp.append(record.replace('\n', '##newline##') + '\n')
        truth_train.append(str(train.loc[i]['gold'])+'\n')
    
    # zero-shot prompting
    zsp = list([no_training.replace('\n', '##newline##') + "\n"])
    truth = list()
    for i in range(0, test.shape[0], 2):
        sentence1 = test.loc[i]['sentence']
        sentence2 = test.loc[i+1]['sentence']
        target = test.loc[i]['lemma']
        record = "Now it's your turn. You have to answer with \"True\" or \"False\"" + '\n' + info.format(sentence1, sentence2, target)
        zsp.append(record.replace('\n', '##newline##') + '\n')
        truth.append(str(test.loc[i]['gold'])+'\n')
    fsp.extend(zsp)
    tp.extend(zsp)

    Path(f'prompt-data/{folder}').mkdir(exist_ok=True, parents=True)
    Path(f'prompt-truth/{folder}').mkdir(exist_ok=True, parents=True)
    
    # store prompts
    for p in ['zsp', 'fsp', 'tp']:
        with open(f'prompt-data/{folder}/{p}.txt', mode='w', encoding='utf-8') as f:
            f.writelines(eval(p))
            
    with open(f'prompt-truth/{folder}/test.txt', mode='w', encoding='utf-8') as f:
        f.writelines(truth)

    with open(f'prompt-truth/{folder}/train.txt', mode='w', encoding='utf-8') as f:
        f.writelines(truth_train)
