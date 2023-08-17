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
