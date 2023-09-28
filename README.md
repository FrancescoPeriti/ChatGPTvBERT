# ChatGPT v BERT: Dawn of Justice for _Semantic Change Detection_
<p align="center">
  <img src="ChatGPTvBERT-meme.png" width="35%" height="35%"/>
</p>

Download data and generate prompts
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

Download data LSC
```
wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip
unzip semeval2020_ulscd_eng.zip
```

# ChatGPT - WebInterface
```bash
python bot4chatgpt.py -d TempoWiC -p ZSp
python bot4chatgpt.py -d TempoWiC -p FSp
python bot4chatgpt.py -d TempoWiC -p MSp
python bot4chatgpt.py -d HistoWiC -p ZSp
python bot4chatgpt.py -d HistoWiC -p FSp
python bot4chatgpt.py -d HistoWiC -p MSp
```

# ChatGPT - API
```bash
python chatgpt-api.py -a your_api -d TempoWiC -p zsp 
python chatgpt-api.py -a your_api -d TempoWiC -p fsp 
python chatgpt-api.py -a your_api -d HistoWiC -p zsp 
python chatgpt-api.py -a your_api -d HistoWiC -p fsp
python chatgpt-api.py -a your_api -d HistoTempoWiC -p zsp  
```

LSC
```bash
python chatgpt-api-LSC.py
```

# BERT
```bash
python store-target-embeddings.py -d data/HistoWiC/ --model bert-base-uncased --batch_size 16 --train_set --test_set --use_gpu
python store-target-embeddings.py -d data/TempoWiC/ --model bert-base-uncased --batch_size 16 --train_set --test_set --use_gpu
python store-target-embeddings.py -d data/HistoTempoWiC/ --model bert-base-uncased --batch_size 16 --train_set --test_set --use_gpu
```

```bash
mv data/HistoWiC/target_embeddings/bert-base-uncased/train/ data/HistoWiC/target_embeddings/bert-base-uncased/dev/
mv data/TempoWiC/target_embeddings/bert-base-uncased/train/ data/TempoWiC/target_embeddings/bert-base-uncased/dev/
mv data/HistoTempoWiC/target_embeddings/bert-base-uncased/train/ data/HistoTempoWiC/target_embeddings/bert-base-uncased/dev/
cp data/TempoWiC/train.txt data/TempoWiC/dev.txt
cp data/HistoWiC/train.txt data/HistoWiC/dev.txt
cp data/HistoTempoWiC/train.txt data/HistoTempoWiC/dev.txt
```

```bash
python bert-wic-stats.py -d data/TempoWiC -m bert-base-uncased --test_set --dev_set
python bert-wic-stats.py -d data/HistoWiC -m bert-base-uncased --test_set --dev_set
python bert-wic-stats.py -d data/HistoTempoWiC -m bert-base-uncased --test_set --dev_set
```
```python
import pandas as pd
pd.read_csv('data/HistoWiC/wic_stats.tsv', sep='\t')
pd.read_csv('data/TempoWiC/wic_stats.tsv', sep='\t')
```
