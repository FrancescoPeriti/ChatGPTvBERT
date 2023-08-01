import os
import csv
import time
import openai
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


# The Answer to the Great Question of Life, the Universe and Everything is Forty-two
random.seed(42)

# Argument parser
parser = argparse.ArgumentParser(prog='ChatGPT prompting', add_help=True)
parser.add_argument('-d', '--dataset',
                    type=str,
                    help='Human annotation dataset to consider')
parser.add_argument('-a', '--api',
                    type=str,
                    default='your_api',
                    help='Your openai api file')
parser.add_argument('-t', '--target',
                    type=str,
                    help='Target word to consider')
parser.add_argument('-o', '--output',
                    type=str,
                    default='result',
                    help='Output folder')
parser.add_argument('-m', '--model',
                    type=str,
                    default='gpt-3.5-turbo',
                    help='Model name')
args = parser.parse_args()


def sample_pair(target, dataset, k=60):
    filter_df_judgments = list()

    # uses.csv
    df_uses = pd.read_csv(f'{dataset}/data/{target}/uses.csv', sep='\t', quoting=csv.QUOTE_NONE)
    contexts = df_uses['context'].values
    groupings = df_uses['grouping'].values
    identifiers = df_uses['identifier'].values

    # local_identifier, integer
    mapping = dict(zip(identifiers, list(range(identifiers.shape[0]))))

    # judgments.csv
    df_judgments = pd.read_csv(f'{dataset}/data/{target}/judgments.csv', sep='\t', quoting=csv.QUOTE_NONE)

    # instances with document of different time periods
    df_instances = df_judgments[['identifier1', 'identifier2']].drop_duplicates()
    instances = [(row['identifier1'], row['identifier2']) for _, row in df_instances.iterrows()
                 if groupings[mapping[row['identifier1']]] != groupings[mapping[row['identifier2']]]]
    instances = random.sample(instances, k)

    # processing for DWUG-English
    target = target.replace('_nn', '').replace('_vb', '')

    # store instances in a new dataframe with their annotations and contexts
    for instance in instances:
        annotations = df_judgments[(df_judgments['identifier1'] == instance[0]) &
                                   (df_judgments['identifier2'] == instance[1])]
        for i, row in annotations.iterrows():
            record = dict()
            record['target'] = target
            record['context1'] = contexts[mapping[instance[0]]]
            record['context2'] = contexts[mapping[instance[1]]]
            record['annotator'] = row['annotator']
            record['judgment'] = row['judgment']
            record['identifier1'] = instance[0]
            record['identifier2'] = instance[1]
            record['local_identifier1'] = mapping[instance[0]]
            record['local_identifier2'] = mapping[instance[1]]
            record['guidelines'] = 'guidelines'

            filter_df_judgments.append(record)

    return pd.DataFrame(filter_df_judgments)

def gpt_lsc(target, sentence1, sentence2, guidelines, model="gpt-4", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": guidelines},
            {"role": "user",
             "content": f"Target word: \"{target}\"\nSentence1: \"{sentence1}\"\nSentence2: \"{sentence2}\""},
        ])

    try:
        return int(response['choices'][0]['message']['content'])
    except:
        if response['choices'][0]['message']['content'] == '-':
            return '-'
        else:
            warnings.warn(f"""The response is not valid. Request: 
            - Target word: {target}
            - Sentence1: {sentence1}
            - Sentence2: {sentence2}
            """)
            return '#'

def test_gpt(instances, guidelines, model, temperature=0):
    new_instances = list()

    bar = tqdm(np.unique(instances.target.values), leave=True, position=0)
    for target in bar:
        df = instances[instances['target'] == target][['context1', 'context2',
                                                       'identifier1', 'identifier2',
                                                       'local_identifier1',
                                                       'local_identifier2']].drop_duplicates().reset_index(drop=True)

        for i, row in df.iterrows():
            # Avoid issues due to too many requests in a short period of time
            time.sleep(10)

            bar.set_description(str(i))
            record = dict()
            record['target'] = target
            record['context1'] = row['context1']
            record['context2'] = row['context2']
            record['annotator'] = model
            record['judgment'] = gpt_lsc(target, row['context1'], row['context2'], guidelines, model, temperature)
            record['identifier1'] = row['identifier1']
            record['identifier2'] = row['identifier2']
            record['local_identifier1'] = row['local_identifier1']
            record['local_identifier2'] = row['local_identifier2']
            record['guidelines'] = guidelines
            new_instances.append(record)

    return pd.DataFrame(new_instances)

# guidelines
long_guidelines = """You will be shown two sentences containing a target word. 
                 Your task is to evaluate how strong the semantic relatedness is 
                 between the two uses of the target word in the two sentences. 

                 Please try to ignore differences between the uses that do not impact their meaning. 
                 For example, 'eat' and 'ate' can express the same meaning, even though one is in present tense, 
                 and the other is in past tense. Also, distinctions between singular and plural 
                 (as in 'carrot' vs. 'carrots') are typically irrelevant for the meaning. 

                 The scale that you will be using for your judgments ranges from 1 (the two uses of the word 
                 have completely unrelated meanings) to 4 (the two uses of the word have identical meanings). 
                 This four-point scale is shown below: 

                 4 Identical; 
                 3 Closely Related; 
                 2 Distantly Related; 
                 1 Unrelated. 

                 Use a dash (i.e. '-') if you can't decide. 

                 Answer only by using '1', '2', '3', '4', or '-'
                """
long_guidelines = " ".join(long_guidelines.split())

short_guidelines = """Evaluate the semantic relatedness between the use of a target word in two 
                 sentences using a scale from 1 to 4: 
                 1 (Unrelated), 
                 2 (Distantly Related), 
                 3 (Closely Related), 
                 4 (Identical).

                 Answer only by using '1', '2', '3', or '4'
                 """
short_guidelines = " ".join(short_guidelines.split())

# arguments
human_annotation_dataset = args.dataset
dataset_name = os.path.basename(os.path.normpath(human_annotation_dataset))
openai.api_key_path = args.api
target = args.target
model = args.model

# sample instances
instances = sample_pair(target, human_annotation_dataset)

# create result folder
Path(f'{args.output}/{dataset_name}/short-guidelines/{model}/').mkdir(parents=True, exist_ok=True)
Path(f'{args.output}/{dataset_name}/long-guidelines/{model}/').mkdir(parents=True, exist_ok=True)

if os.path.exists(f'{args.output}/{dataset_name}/short-guidelines/{model}/{target.replace("_nn", "").replace("_vb", "")}.csv'):
    pass
else:
    result_short_guidelines = test_gpt(instances, short_guidelines, model, 0)
    result_short_guidelines = pd.concat([instances, result_short_guidelines])
    result_short_guidelines.to_csv(f'{args.output}/{dataset_name}/short-guidelines/{model}/{target.replace("_nn", "").replace("_vb", "")}.csv', index=False, sep='\t')

# Avoid issues due to too many requests in a short period of time
time.sleep(60*3)

if os.path.exists(f'{args.output}/{dataset_name}/long-guidelines//{target.replace("_nn", "").replace("_vb", "")}.csv'):
    pass
else:
    result_long_guidelines = test_gpt(instances, long_guidelines, model, 0)
    result_long_guidelines = pd.concat([instances, result_long_guidelines])
    result_long_guidelines.to_csv(f'{args.output}/{dataset_name}/long-guidelines/{model}/{target.replace("_nn", "").replace("_vb", "")}.csv', index=False, sep='\t')
