# (Chat)GPT v BERT: Dawn of Justice for _Semantic Change Detection_
This is the official repository for our paper **(Chat)GPT v BERT: Dawn of Justice for _Semantic Change Detection_**

<p align="center">
  <img src="ChatGPTvBERT-meme.png" width="35%" height="35%"/>
  <figcaption>Figure 1: The title of this paper draws inspiration by the movie <i>Batman v Superman: Dawn of Justice</i>. In this paper, we leverage the analogy of (Chat)GPT and BERT, powerful and popular PFMs, as two <i>lexical superheroes</i> often erroneously associated for solving similar problems.  Our
aim is to shed lights on the potential of (Chat)GPT for semantic change detection.</figcaption>
</p>

## Table of Contents

- [Abstract](#abstract)
- [ChatGPT Conversations](#chatgpt-conversations)
- [Getting Started](#getting-started)
- [Reproducing Results](#reproducing-results)
- [References](#references)

## Abstract
In the universe of Natural Language Processing, Transformer-based language models like BERT and (Chat)GPT have emerged as lexical superheroes with great power to solve open research problems. In this paper, we specifically focus on the temporal problem of semantic change, and evaluate their ability to solve two diachronic extensions of the Word-in-Context (WiC) task: TempoWiC and HistoWiC. In particular, we investigate the potential of a novel, off-the-shelf technology like ChatGPT (and GPT) 3.5 compared to BERT, which represents a family of models that currently stand as the state-of-the-art for modeling semantic change. Our experiments represent the first attempt to assess the use of (Chat)GPT for studying semantic change. Our results indicate that ChatGPT performs significantly worse than the foundational GPT version. Furthermore, our results demonstrate that (Chat)GPT achieves slightly lower performance than BERT in detecting long-term changes but performs significantly worse in detecting short-term changes.

## ChatGPT Conversations
To access the answers generated by ChatGPT for multiple experiments, run these commands:

```bash
unzip chatgpt-conversations.zip
mv chatgpt-conversations/* .
unzip dump-web-chat.zip
```
<p><b> chatgpt-conversations </b></p>
The <i>chatgpt-conversations</i> folder contains 10 sub-folders named <i>chat-conversation{i}</i> for each experiment run, where <i>i</i> ranges from 1 to 10. Within each <i>chat-conversation{i}</i> sub-folder, you will find the following three sub-folders:


- <i>HistoWiC</i>: it contains ChatGPT conversations related to the experiments on HistoWiC.
- <i>TempoWiC</i>: it contains ChatGPT conversations related to the experiments on TempoWiC.
- <i>LSC</i>: it contains ChatGPT conversations related to the experiments on LSC

Additionally, both <i>HistoWiC</i> and <i>TempoWiC</i> have two sub-folders named <i>zsp</i> and <i>fsp</i>, corresponding to Zero-shot prompting and Few-shot prompting, respectively. The <i>LSC</i> sub-folder contains a sub-folder called <i>graded</i>. Each of these sub-folders (i.e. zsp, fsp, graded) contains files corresponding to specific temperatures experimented during the respective run (e.g., <i>0.0.json</i>, ..., <i>2.0.json</i>).

Navigate through the folders to access the data related to each experiment run and its corresponding temperature values.

- chat-conversation1
  - HistoWiC
    - zsp
      - 0.0.json
      - ...
      - 2.0.json
    - fsp
      - 0.0.json
      - ...
      - 2.0.json
  - TempoWiC
    - zsp
      - 0.0.json
      - ...
      - 2.0.json
    - fsp
      - 0.0.json
      - ...
      - 2.0.json
  - LSC
    - graded
      - 0.0.json
      - ...
      - 2.0.json
  (Repeat for chat-conversation2 to chat-conversation10)

<p><b> dump-web-chat </b></p>
The <i>dump-web-chat</i> folder contains a dump of our ChatGPT Web conversations (<a href="https://chat.openai.com/">ChatGPT</a> -> Setting -> Data controls -> Export data).

## Getting Started
Before you begin, ensure you have met the following requirements:

- Python 3.8+
- Required Python packages (listed in `requirements.txt`)

To install the required packages, you can use pip:

```bash
pip install -r requirements.txt
```
## Reproducing Results
<b>Data</b>

- Download data and generate prompts
```bash
python download-histowic.py
python download-tempowic.py
python generate-prompts.py

mkdir prompt-data/HistoTempoWiC
cat prompt-data/TempoWiC/zsp.txt > prompt-data/HistoTempoWiC/zsp.txt
tail -n+2 prompt-data/HistoWiC/zsp.txt >> prompt-data/HistoTempoWiC/zsp.txt
mkdir prompt-data/HistoTempoWiC
cat prompt-truth/TempoWiC/test.txt > prompt-truth/HistoTempoWiC/test.txt
cat prompt-truth/HistoWiC/test.txt >> prompt-truth/HistoTempoWiC/test.txt
cat prompt-truth/TempoWiC/train.txt > prompt-truth/HistoTempoWiC/train.txt
cat prompt-truth/HistoWiC/train.txt >> prompt-truth/HistoTempoWiC/train.txt

mkdir data/HistoTempoWiC
cat data/TempoWiC/test.txt > data/HistoTempoWiC/test.txt
cat data/HistoWiC/test.txt >> data/HistoTempoWiC/test.txt
cat data/TempoWiC/train.txt > data/HistoTempoWiC/train.txt
cat data/HistoWiC/train.txt >> data/HistoTempoWiC/train.txt
```

- Download data for Lexical Semantic Change detection (LSC)
```
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip
unzip semeval2020_ulscd_eng.zip
```

<b>ChatGPT - WebInterface</b>

- Utilize the <i>bot4chatgpt</i> bot to chat with ChatGPT through the OpenAI GUI. Follow the instructions provided by the script.

```bash
python bot4chatgpt.py -d TempoWiC -p ZSp
python bot4chatgpt.py -d TempoWiC -p FSp
python bot4chatgpt.py -d TempoWiC -p MSp
python bot4chatgpt.py -d HistoWiC -p ZSp
python bot4chatgpt.py -d HistoWiC -p FSp
python bot4chatgpt.py -d HistoWiC -p MSp
```

<b>GPT - API</b>

- Create a file named 'your_api' containing your OpenAI API token.
- Chat with ChatGPT through the OpenAI API using various prompts and temperature settings. Execute the following commands (each run will test different temperature values):
  
```bash
python chatgpt-api.py -a your_api -d TempoWiC -p zsp 
python chatgpt-api.py -a your_api -d TempoWiC -p fsp 
python chatgpt-api.py -a your_api -d HistoWiC -p zsp 
python chatgpt-api.py -a your_api -d HistoWiC -p fsp
python chatgpt-api.py -a your_api -d HistoTempoWiC -p zsp  
```

<b>Lexical Semantic Change (LSC)</b>

- Test the knowledge of ChatGPT on historical semantic changes.
```bash
python chatgpt-api-LSC.py
```

<b>BERT</b>

- Extract embeddings
```bash
python store-target-embeddings.py -d data/HistoWiC/ --model bert-base-uncased --batch_size 16 --train_set --test_set --use_gpu
python store-target-embeddings.py -d data/TempoWiC/ --model bert-base-uncased --batch_size 16 --train_set --test_set --use_gpu
python store-target-embeddings.py -d data/HistoTempoWiC/ --model bert-base-uncased --batch_size 16 --train_set --test_set --use_gpu
```

- Run the following commands to use Train as Dev set (to find optimal threshold)
```bash
mv data/HistoWiC/target_embeddings/bert-base-uncased/train/ data/HistoWiC/target_embeddings/bert-base-uncased/dev/
mv data/TempoWiC/target_embeddings/bert-base-uncased/train/ data/TempoWiC/target_embeddings/bert-base-uncased/dev/
mv data/HistoTempoWiC/target_embeddings/bert-base-uncased/train/ data/HistoTempoWiC/target_embeddings/bert-base-uncased/dev/
cp data/TempoWiC/train.txt data/TempoWiC/dev.txt
cp data/HistoWiC/train.txt data/HistoWiC/dev.txt
cp data/HistoTempoWiC/train.txt data/HistoTempoWiC/dev.txt
```
- Compute BERT stats on Test set
```bash
python bert-wic-stats.py -d data/TempoWiC -m bert-base-uncased --test_set --dev_set
python bert-wic-stats.py -d data/HistoWiC -m bert-base-uncased --test_set --dev_set
python bert-wic-stats.py -d data/HistoTempoWiC -m bert-base-uncased --test_set --dev_set
```

- Explore statistics
```python
import pandas as pd
pd.read_csv('data/HistoWiC/wic_stats.tsv', sep='\t')
pd.read_csv('data/TempoWiC/wic_stats.tsv', sep='\t')
```

<b>Plots</b>

- Run the <i>ChatGPTvBERT.ipynb</i> notebook.

### References

Coming....
```
@inproceedings{periti2022chatgPT,
    title = {{ChatGPT v BERT: Dawn of Justice for Semantic Change Detection}},
    author = "Periti, Francesco  and
              Dubossarsky, Haim  and
              Tahmasebi, Nina",
    booktitle = "Proceedings of [...]",
    month = [...],
    year = "[...]",
    address = "[...]",
    publisher = "Association for Computational Linguistics",
    url = "[...]",
    doi = "[...]",
    pages = "[...]",
    abstract = "[...]",
}
```
Temporary...
@Misc{periti2024chatgpt,
  title = {{(Chat)GPT v BERT: Dawn of Justice for Semantic Change Detection}}, 
  author = {Periti, Francesco and Dubossarsky, Haim and Tahmasebi, Nina},
  year = {2024},
  eprint = {2401.14040},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2401.14040}
}


```
@Misc{montanelli2023survey,
  title = {{A Survey on Contextualised Semantic Shift Detection}}, 
  author = {Stefano Montanelli and Francesco Periti},
  year = {2023},
  eprint = {2304.01666},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2304.01666}
}
```
