---
language: en
datasets:
- superb
tags:
- speech
- audio
- hubert
- audio-classification
license: apache-2.0
---

# Hubert-Large for Emotion Recognition

## Model description

This is a ported version of 
[S3PRL's Hubert for the SUPERB Emotion Recognition task](https://github.com/s3prl/s3prl/tree/master/s3prl/downstream/emotion).

The base model is [hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k), which is pretrained on 16kHz 
sampled speech audio. When using the model make sure that your speech input is also sampled at 16Khz. 

For more information refer to [SUPERB: Speech processing Universal PERformance Benchmark](https://arxiv.org/abs/2105.01051)

## Task and dataset description

Emotion Recognition (ER) predicts an emotion class for each utterance. The most widely used ER dataset
[IEMOCAP](https://sail.usc.edu/iemocap/) is adopted, and we follow the conventional evaluation protocol: 
we drop the unbalanced emotion classes to leave the final four classes with a similar amount of data points and 
cross-validate on five folds of the standard splits.

For the original model's training and evaluation instructions refer to the 
[S3PRL downstream task README](https://github.com/s3prl/s3prl/tree/master/s3prl/downstream#er-emotion-recognition).


## Usage examples

You can use the model via the Audio Classification pipeline:
```python
from datasets import load_dataset
from transformers import pipeline

dataset = load_dataset("anton-l/superb_demo", "er", split="session1")

classifier = pipeline("audio-classification", model="superb/hubert-large-superb-er")
labels = classifier(dataset[0]["file"], top_k=5)
```

Or use the model directly:
```python
import torch
import librosa
from datasets import load_dataset
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

def map_to_array(example):
    speech, _ = librosa.load(example["file"], sr=16000, mono=True)
    example["speech"] = speech
    return example

# load a demo dataset and read audio files
dataset = load_dataset("anton-l/superb_demo", "er", split="session1")
dataset = dataset.map(map_to_array)

model = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")

# compute attention masks and normalize the waveform if needed
inputs = feature_extractor(dataset[:4]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")

logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)
labels = [model.config.id2label[_id] for _id in predicted_ids.tolist()]
```

## Eval results

The evaluation metric is accuracy.

|        | **s3prl** | **transformers** |
|--------|-----------|------------------|
|**session1**| `0.6762`  | `N/A`         |

### BibTeX entry and citation info

```bibtex
@article{yang2021superb,
  title={SUPERB: Speech processing Universal PERformance Benchmark},
  author={Yang, Shu-wen and Chi, Po-Han and Chuang, Yung-Sung and Lai, Cheng-I Jeff and Lakhotia, Kushal and Lin, Yist Y and Liu, Andy T and Shi, Jiatong and Chang, Xuankai and Lin, Guan-Ting and others},
  journal={arXiv preprint arXiv:2105.01051},
  year={2021}
}
```