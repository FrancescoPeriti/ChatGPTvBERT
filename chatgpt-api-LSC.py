import os
import time
import json
import openai
import random
import argparse
import  numpy as np
from tqdm import tqdm
from pathlib import Path

# The Answer to the Great Question of Life, the Universe and Everything is Forty-two
random.seed(42)


def wait_random_time(min_=60, max_=75):
    # There is no hourly usage limit for ChatGPT,
    # but each response is subject to a word and
    # character limit of approximately 500
    # words or 4000 characters
    return random.choice(list(range(min_, max_)))

# this is the same as ChatGPT Web
start_content = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.\nKnowledge cutoff: 2021-09\nCurrent date: 2023-04-16"
model = 'gpt-3.5-turbo'


# You have to send the entire conversation back to the API each time.
# ChatGPT does not remember conversations, it is just sending what is
# in the chat window back to the API every time you hit submit.
def input_prompt(messages, temperature):
    chatgpt = openai.ChatCompletion.create(model=model, temperature=temperature, messages=messages)
    return chatgpt['choices'][0]['message']['content']

def check_end(dataset, prompt):
    for temperature in np.arange(0, 2.2, 0.2):
        if not os.path.exists(f'chatgpt-conversations/{dataset}/{prompt}/{temperature}.json'):
            return False

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ChatGPT prompting', add_help=True)
    parser.add_argument('-f')
    parser.add_argument('-a', '--api',
                        type=str,
                        default='your_api',
                        help='Your openai api file')
    args = parser.parse_args()

    openai.api_key_path = args.api

    # set sliding window boundaries
    first_message = {"role": "system", "content": start_content}

    while not check_end(args.dataset, args.prompt):
        for temperature in np.arange(0, 2.2, 0.2):
            temperature = round(temperature, 1)
            if os.path.exists(f'chatgpt-conversations/LSC/{temperature}.json'):
                continue

            history = [first_message]
            gold_count = 0
            with open(f'semeval2020_ulscd_eng/targets.txt', mode='r', encoding='utf-8') as f:
                for line in tqdm(list(f)):
                    target = line.replace('##newline##', '\n').replace('_vb', '').replace('_nn', '')
                    line = f"""Consider the following two time periods and target word. How much has the meaning of the target word changed between the two periods? Rate the lexical semantic change on a scale from 0 to 1. Provide only a score.\nTarget: {target}\nTime period 1: 1810–1860\nTime period 2: 1960–2010"""
                    tmp = {"role": "user", "content": line}
                    history.append(tmp)

                    tmp_history = [first_message, tmp]

                    error = True
                    while error:
                        try:
                            content_ = input_prompt(tmp_history, temperature)
                            wait_random_time()
                        except:
                            continue
                        error = False
                    answer = {"role": "assistant", "content": content_}
                    history.append(answer)

                    gold_count += 1

                    # avoid rate limit of tokens per min. Limit: 90000 / min
                    if len(history) % 45 == 0:
                        time.sleep(wait_random_time())


            Path(f'chatgpt-conversations/LSC/').mkdir(exist_ok=True, parents=True)
            with open(f'chatgpt-conversations/LSC/{temperature}.json', mode='w',
                      encoding='utf-8') as f:
                json.dump(history, f)
                print('Done ', temperature)
                time.sleep(wait_random_time())
