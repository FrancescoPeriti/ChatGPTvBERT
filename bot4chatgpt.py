import time
import random
import argparse
import pyautogui
import pyperclip

# The Answer to the Great Question of Life, the Universe and Everything is Forty-two
random.seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Bot interacting with ChatGPT demo', add_help=True)
    parser.add_argument('-d', '--dataset',
                        type=str,
                        help='Specifies the dataset for the analysis.')
    parser.add_argument('-p', '--prompt',
                        type=str,
                        help='Specifies the prompt for the analysis.')
    args = parser.parse_args()

    print("The script is designed to control both your keyboard and mouse, "
          "thereby emulating human queries to interact with ChatGPT. "
          "Thus, it's important that you refrain from using the computer "
          "while the script is running. Do you agree with this? [Yes, No]")
    answer = input()
    answer = 'n' if len(answer) == 0 else answer
    
    if answer[0].lower() != 'y':
        print("I'm sorry, you should agree to continue")
        exit(0)
    
    answer = 'q'
    while answer[0].lower() != 'y':
        print("You will need to open a web browser where ChatGPT is running. "
              "Create a new empty chat. "
              "Place your mouse cursor at the message typing bar when the first sound begins, "
              "and keep it there until the second sound (around 10 seconds). "
              "This is required to calibrate the position of the text bar. "
              "Then, move your mouse cursor to the send message button to the right and hold it there "
              "from the second sound to the third sound (around 10 seconds). "
              "This is required to calibrate the position of the send message button. "
              "Then you should retain to use your computer till the end of the processing. "
              "The script will start automatically. "
              "If there is no activity observed for a continuous duration of 5 minutes, then the execution ended."
              "Are you ready? [Yes/No] Type q if you prefer to quit")
    
        answer = input()
        answer = 'n' if len(answer) == 0 else answer
    
        if answer == 'q':
            exit(0)
    
    print('\a') # 1st noise sound
    time.sleep(6)
    pos_text = pyautogui.position()
    time.sleep(4) # 2nd noise sound
    print('\a')
    time.sleep(6) # 3rd noise sound
    pos_send = pyautogui.position()
    time.sleep(4)
    print('\a') # 4th noise sound
    
    def wait_random_time(min_=45, max_=75):
        # There is no hourly usage limit for ChatGPT,
        # but each response is subject to a word and
        # character limit of approximately 500
        # words or 4000 characters
        return random.choice(list(range(min_, max_)))
    
    
    with open(f'prompt-data/{args.dataset}/{args.prompt}.txt', mode='r', encoding='utf-8') as f:
        for line in f:
            # click on the text bar
            pyautogui.click(pos_text.x, pos_text.y, clicks=2)
    
            # wait a little bit
            time.sleep(wait_random_time())
    
            # copy the text
            pyperclip.copy(line.replace('##newline##', '\n').replace('_vb', '').replace('_nn', ''))
    
            # past the text into the text bar
            pyautogui.hotkey('ctrl', 'v', interval=0.25)
    
            # wait a little bit
            time.sleep(wait_random_time())
    
            # send the query
            pyautogui.click(pos_send.x, pos_send.y, clicks=2)
    
            # wait additional 3 arbitrary second
            time.sleep(3)
    
    print('END')
    print('\a') # noise sound
