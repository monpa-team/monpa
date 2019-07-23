# 罔拍 MONPA: Multi-Objective NER POS Annotator 

## Python Package Project

MONPA（罔拍）是一個提供正體中文分詞及 POS, NE 標註的模型。初期只有網站版本（<http://monpa.iis.sinica.edu.tw:9000/chunk>），本計劃將把 monpa 包裝成可以 pip install 的 python package。

最新版的 monpa model 是使用 (py)torch 1.0 框架訓練出來的模型，所以在使用本版本前，請先安裝 torch 1.* 以上版本才能正常使用 monpa 套件。

<span style="color:red"> **注意：** </span>

1. 建議以原文輸入 monpa 完成切詞後，再視需求濾掉停留字（stopword）及標點符號（punctuation）。
2. 每次輸入予 monpa 做切詞的原文超過 140 字元的部分將被截斷丟失，建議先完成合適斷句後再應用 monpa 切詞。

## 安裝 (py)torch 1.* 套件

請至 https://pytorch.org/ 依照頁面指示點選你的作業系統（Your OS），想要安裝的方式（Package），使用的 python 版本（Language），如有 GPU 就選擇合適的 CUDA 版本。最後在 『Run this Command: 』就會出現建議執行的安裝指令。以下以 Mac OS X 舉例說明之：

#### Mac OS X

##### 使用 conda 安裝

在終端機命令列，輸入命令：

```bash
conda install pytorch torchvision -c pytorch
```

確認是否安裝成功及能呼叫該套件，在命令列輸入 python 進入 python 對談式介面。

```bash
python
```

在對談式介面如下操作，

```python
>>> import torch
```

```python
>>> print(torch.__version__)
```

應該要看到顯示的版本是 1.1.*，表示有安裝成功並可呼叫。輸入 exit()，跳出 python 交談式介面。

```python
>>> exit()
```

##### 使用 pip 安裝

在終端機命令列，輸入命令：

```bash
pip3 install torch torchvision
```

確認是否安裝成功及能呼叫該套件，在命令列輸入 python 進入 python 對談式介面。

```bash
python
```

在對談式介面如下操作，

```python
>>> import torch
```

```python
>>> print(torch.__version__)

```

應該要看到顯示的版本是 1.1.*，表示有安裝成功並可呼叫。輸入 exit()，跳出 python 交談式介面。

```python
>>> exit()

```

#### Linux

步驟同 Mac OS X，注意更換 Your OS 及 CUDA 的選擇。

#### Windows

步驟同 Mac OS X，注意更換 Your OS 及 CUDA 的選擇。譬如下例：

## 安裝 monpa 套件

monpa 已經支援直接使用 pip 指令安裝，各作業系統的安裝步驟都相同。

```bash
pip install monpa

```

沒有錯誤訊息，就是好消息。

確認是否安裝成功及能呼叫該套件，在命令列輸入 python 進入 python 對談式介面。

```bash
python

```

在對談式介面如下操作，

```python
>>> import monpa

```

```python
>>> print(monpa.__version__)

```

可以看到版本顯示，表示有安裝成功並可呼叫。輸入 exit()，跳出 python 交談式介面。

```python
>>> exit()

```

## 使用 monpa 的簡單範例

Mac OS X, Linux 及 Windows 都相同，先啟動 jupyter。

```bash
jupyter notebook

```

再來就是你熟知的 jupyter notebook 介面，開啟一個 python kernel notebook，並引入 monpa package。

```python
import monpa

```

沒有錯誤訊息，就是好消息。

### cut function

若只需要中文分詞結果，請用 cut function，回傳值是 list 格式。簡單範例如下：

```python
sentence = "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。"
result = monpa.cut(sentence)

for t in result:
  print(t)

```

輸出

```python
蔡英文
總統
今天
受
邀
參加
台北市政府
所
舉辦
的
陽明山
馬拉松
比賽
。

```

### pseg function

若需要中文分詞及其 POS 結果，請用 pseg function，回傳值是 list of list 格式，簡單範例如下：

```python
sentence = "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。"
result = monpa.pseg(sentence)

for t in result:
  print(t)

```

輸出

```python
['蔡英文', 'per']
['總統', 'na']
['今天', 'nd']
['受', 'p']
['邀', 'vf']
['參加', 'vc']
['台北市政府', 'org']
['所', 'd']
['舉辦', 'vc']
['的', 'de']
['陽明山', 'loc']
['馬拉松', 'na']
['比賽', 'na']
['。', 'periodcategory']

```

### 載入自訂詞典 load_userdict function

如果需要自訂詞典，請依下列格式製作詞典文字檔，再使用此功能載入。簡單範例如下：

假設製作一個 userdict.txt 檔，每行含三部分，必須用『空格 （space）』隔開，依次是：詞語、詞頻（請填數值，目前無作用）、詞性（未能確定，請填 NER），順序不可錯亂。

**注意：最後不要留空行或任何空白空間。***

```reStructuredText
台北市政府 100 ner
受邀 100 v

```

當要使用自訂詞時，請於執行分詞前先做 load_userdict，將自訂詞典載入到 monpa 模組。

請將本範例的 『 ./userdict.txt 』改成實際放置自訂詞文字檔路徑及檔名。

```python
monpa.load_userdict("./userdict.txt")

```

延用前例，用 pseg function，可發現回傳值已依自訂詞典分詞，譬如『受邀』為一個詞而非先前的兩字分列輸出，『台北市政府』也依自訂詞輸出。

```python
sentence = "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。"
result = monpa.pseg(sentence)

for t in result:
  print(t)

```

輸出

```python
['蔡英文', 'per']
['總統', 'na']
['今天', 'nd']
['受邀', 'v']
['參加', 'vc']
['台北市政府', 'ner']
['所', 'd']
['舉辦', 'vc']
['的', 'de']
['陽明山', 'loc']
['馬拉松', 'loc']
['比賽', 'na']
['。', 'periodcategory']

```

## 其他

See our paper [MONPA: Multi-objective Named-entity and Part-of-speech Annotator for Chinese using Recurrent Neural Network](https://www.aclweb.org/anthology/papers/I/I17/I17-2014/) for more information about the model detail.

##### Abstract

Part-of-speech (POS) tagging and named entity recognition (NER) are crucial steps in natural language processing. In addition, the difficulty of word segmentation places additional burden on those who intend to deal with languages such as Chinese, and pipelined systems often suffer from error propagation. This work proposes an end-to-end model using character-based recurrent neural network (RNN) to jointly accomplish segmentation, POS tagging and NER of a Chinese sentence. Experiments on previous word segmentation and NER datasets show that a single model with the proposed architecture is comparable to those trained specifically for each task, and outperforms freely-available softwares. Moreover, we provide a web-based interface for the public to easily access this resource.

#### Citation:

##### APA:

Hsieh, Y. L., Chang, Y. C., Huang, Y. J., Yeh, S. H., Chen, C. H., & Hsu, W. L. (2017, November). MONPA: Multi-objective Named-entity and Part-of-speech Annotator for Chinese using Recurrent Neural Network. In *Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 2: Short Papers)* (pp. 80-85).

##### BibTex

```text
@inproceedings{hsieh-etal-2017-monpa,
    title = "{MONPA}: Multi-objective Named-entity and Part-of-speech Annotator for {C}hinese using Recurrent Neural Network",
    author = "Hsieh, Yu-Lun  and
      Chang, Yung-Chun  and
      Huang, Yi-Jie  and
      Yeh, Shu-Hao  and
      Chen, Chun-Hung  and
      Hsu, Wen-Lian",
    booktitle = "Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = nov,
    year = "2017",
    address = "Taipei, Taiwan",
    publisher = "Asian Federation of Natural Language Processing",
    url = "https://www.aclweb.org/anthology/I17-2014",
    pages = "80--85",
    abstract = "Part-of-speech (POS) tagging and named entity recognition (NER) are crucial steps in natural language processing. In addition, the difficulty of word segmentation places additional burden on those who intend to deal with languages such as Chinese, and pipelined systems often suffer from error propagation. This work proposes an end-to-end model using character-based recurrent neural network (RNN) to jointly accomplish segmentation, POS tagging and NER of a Chinese sentence. Experiments on previous word segmentation and NER datasets show that a single model with the proposed architecture is comparable to those trained specifically for each task, and outperforms freely-available softwares. Moreover, we provide a web-based interface for the public to easily access this resource.",
}

```
