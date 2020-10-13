
# g2pM
[![Release](https://img.shields.io/badge/release-v0.1.2.5-green)](https://pypi.org/project/g2pM/)
[![Downloads](https://pepy.tech/badge/g2pm)](https://pepy.tech/project/g2pm)
[![license](https://img.shields.io/badge/license-Apache%202.0-red)](https://github.com/kakaobrain/g2pM/blob/master/LICENSE)

This is the official repository of our paper [A Neural Grapheme-to-Phoneme Conversion Package for MandarinChinese Based on a New Open Benchmark Dataset](https://arxiv.org/abs/2004.03136) (**Interspeech 2020**).

## Install
```
pip install g2pM
```

## The CPP Dataset
In data folder, there are [train/dev/test].sent files and [train/dev/test].lb files. In *.sent file, each lines corresponds to one sentence and a special symbol ▁ (U+2581) is added to the left and right of polyphonic character. The pronunciation of the corresponding character is at the same line from *.lb file. For each sentence, there could be several polyphonic characters, but we randomly choose only one polyphonic character to annotate.

## Requirements
* python >= 3.6
* numpy

## Usage
If you want to remove all the digits which denote the tones, set tone=False. Default setting is tone=True. <br />
If you want to split all the non Chinese characters (e.g. digit), set char_split=True. Default setting is char_split=False. <br />

```
>>> from g2pM import G2pM
>>> model = G2pM()
>>> sentence = "然而，他红了20年以后，他竟退出了大家的视线。"
>>> model(sentence, tone=True, char_split=False)
['ran2', 'er2', '，', 'ta1', 'hong2', 'le5', '20', 'nian2', 'yi3', 'hou4', '，', 'ta1', 'jing4', 'tui4', 'chu1', 'le5', 'da4', 'jia1', 'de5', 'shi4', 'xian4', '。']
>>> model(sentence, tone=False, char_split=False)
['ran', 'er', '，', 'ta', 'hong', 'le', '2', '0', 'nian', 'yi', 'hou', '，', 'ta', 'jing', 'tui', 'chu', 'le', 'da', 'jia', 'de', 'shi', 'xian', '。']
>>> model(sentence, tone=True, char_split=True)
['ran2', 'er2', '，', 'ta1', 'hong2', 'le5', '2', '0', 'nian2', 'yi3', 'hou4', '，', 'ta1', 'jing4', 'tui4', 'chu1', 'le5', 'da4', 'jia1', 'de5', 'shi4', 'xian4', '。']

```

## Model Size
| Layer                 | Size    |
|-----------------------|---------|
| Embedding             | 64      |
| LSTM x1               | 64      |
| Fully-Connected x2    | 64      |
| Total # of parameters | 477,228 |
| Model size            | 1.7MB   |
| Package size          | 2.1MB   |

## Evaluation Result

| Model            | Dev.            | Test         |
| :--------------| --------------: |:--------------:|
| g2pC                 | 84.84                | 84.45           |
| xpinyin(0.5.6)       | 78.74                | 78.56           |
| pypinyin(0.36.0)     | 85.44                | 86.13           |
| Majority Vote        | 92.15                | 92.08           |
| Chinese Bert         | **97.95**            | **97.85**       |
| Ours                 | 97.36                | 97.31           |


## Reference
To cite the code/data/paper, please use this BibTex
```bibtex
@article{park2020g2pm,
 author={Park, Kyubyong and Lee, Seanie},
 title = {A Neural Grapheme-to-Phoneme Conversion Package for Mandarin Chinese Based on a New Open Benchmark Dataset
},
 journal={Proc. Interspeech 2020},
 url = {https://arxiv.org/abs/2004.03136},
 year = {2020}
}
```
