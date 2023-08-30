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
        temperature = round(temperature, 1)
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
    parser.add_argument('-d', '--dataset',
                        type=str,
                        default='TempoWiC',
                        help='Specifies the dataset for the analysis')
    parser.add_argument('-p', '--prompt',
                        type=str,
                        default='zsp',
                        help='Specifies the prompt for the analysis')
    parser.add_argument('-o', '--output',
                        type=str,
                        default='result',
                        help='Output folder')
    args = parser.parse_args()

    openai.api_key_path = args.api

    # set sliding window boundaries
    if args.prompt == 'zsp':
        end, start = 1, -30
    elif args.prompt == 'fsp':
        end, start = 26, -5

    first_message = {"role": "system", "content": start_content}

    with open(f'prompt-truth/{args.dataset}/truth.txt', mode='r', encoding='utf-8') as f:
        gold = [bool(int(g)) for g in f.readlines()]

    while not check_end(args.dataset, args.prompt):
        for temperature in list(np.arange(0, 2.2, 0.2))[::-1]:
            temperature = round(temperature, 1)
            if os.path.exists(f'chatgpt-conversations/{args.dataset}/{args.prompt}/{temperature}.json'):
                continue
            try:
                history = [first_message]
                gold_count = 0
                with open(f'prompt-data/{args.dataset}/{args.prompt}.txt', mode='r', encoding='utf-8') as f:
                    for line in tqdm(list(f)):
                        line = line.replace('##newline##', '\n').replace('_vb', '').replace('_nn', '')
                        tmp = {"role": "user", "content": line}
                        history.append(tmp)

                        # if tmp != nnnn[-1]:
                        #    continue
                        # history = nnnn

                        # avoid model's maximum context length is 4097 tokens
                        tmp_history = history[:end] + history[start:]

                        answer = {"role": "assistant", "content": input_prompt(tmp_history, temperature)}
                        history.append(answer)

                        if line.startswith("Now it's your turn"):
                            gold_count += 1

                        # avoid rate limit of tokens per min. Limit: 90000 / min
                        if len(history) % 45 == 0:
                            time.sleep(wait_random_time())


                Path(f'chatgpt-conversations/{args.dataset}/{args.prompt}/').mkdir(exist_ok=True, parents=True)
                with open(f'chatgpt-conversations/{args.dataset}/{args.prompt}/{temperature}.json', mode='w',
                          encoding='utf-8') as f:
                    json.dump(history, f)
                    print('Done ', temperature)
            except:
                print('Fail ', temperature)
                continue
