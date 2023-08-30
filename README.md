Download data and generate prompts
```bash
python download-histowic.py
python download-tempowic.py
python generate-prompts.py
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
python chatgpt-api.py -a your_api -d TempoWiC -p zsp -o result
python chatgpt-api.py -a your_api -d TempoWiC -p fsp -o result
python chatgpt-api.py -a your_api -d HistoWiC -p zsp -o result
python chatgpt-api.py -a your_api -d HistoWiC -p fsp -o result
```

# BERT
```bash
python store-target-embeddings.py -d data/HistoWiC/ --model bert-base-uncased --batch_size 16 --train_set --test_set --use_gpu
python store-target-embeddings.py -d data/TempoWiC/ --model bert-base-uncased --batch_size 16 --train_set --test_set --use_gpu
```

```bash
mv data/HistoWiC/target_embeddings/bert-base-uncased/train/ data/HistoWiC/target_embeddings/bert-base-uncased/dev/
mv data/TempoWiC/target_embeddings/bert-base-uncased/train/ data/TempoWiC/target_embeddings/bert-base-uncased/dev/
cp data/TempoWiC/train.txt data/TempoWiC/dev.txt
cp data/HistoWiC/train.txt data/HistoWiC/dev.txt
```

```bash
python bert-wic-stats.py -d data/TempoWiC -m bert-base-uncased --test_set --dev_set
python bert-wic-stats.py -d data/HistoWiC -m bert-base-uncased --test_set --dev_set
```
```python
import pandas as pd
pd.read_csv('data/HistoWiC/wic_stats.tsv', sep='\t')
pd.read_csv('data/TempoWiC/wic_stats.tsv', sep='\t')
```
