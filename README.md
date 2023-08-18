Download data and generate prompts
```bash
python download-histowic.py
python download-tempowic.py
python generate-prompts.py
```

# ChatGPT - WebInterface
```bash
python bot4chatgpt.py -d [dataset] -p [prompt]
```
where
```
-d --dataset [dataset]
    Specifies the dataset for the analysis.

    Options:
        TempoWiC - Temporal Word-in-Context
        HistoWiC - Historical Word-in-Context

-p --prompt [prompt]
    Specifies the prompt for the analysis.

    Options:
        zsp - Zero-shot prompting
        fsp - Few-shot prompting
        tp - Trained prompting
```

# ChatGPT - API

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
