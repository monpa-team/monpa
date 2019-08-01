# 罔拍 MONPA: Multi-Objective NER POS Annotator

MONPA（罔拍）是一個提供正體中文分詞及 POS, NE 標註的模型。初期只有使用原始模型（v0.1）的網站版本（<http://monpa.iis.sinica.edu.tw:9000/chunk>），本計劃將把新版 monpa (v0.2) 包裝成可以 pip install 的 python package。(*提醒：因網站版為 v0.1，與 python 套件版 v0.2 以上的分詞結果可能不同。*)

最新版的 monpa model 是使用 pytorch 1.0 框架訓練出來的模型，所以在使用本版本前，請先安裝 torch 1.* 以上版本才能正常使用 monpa 套件。

**注意：**

1. 建議以原文輸入 monpa 完成切詞後，再視需求濾掉停留字（stopword）及標點符號（punctuation）。
2. 每次輸入予 monpa 做切詞的原文超過 140 字元的部分將被截斷丟失，建議先完成合適斷句後再應用 monpa 切詞。可參考 wiki [如何將長文切成短句再用 monpa 切詞？](https://github.com/monpa-team/monpa/wiki/Example-1：將長句處理成短句再運用-monpa-完成分詞)）
3. 支援 python >= 3.6，不支援 python 2.x。

## 安裝 monpa 套件

monpa 已經支援直接使用 pip 指令安裝，各作業系統的安裝步驟都相同。

```bash
pip install monpa
```

安裝時將自動檢查有無 torch >= 1.0 及 requests 等套件，若無則由 pip 直接安裝。

## 使用 monpa 的簡單範例

引入 monpa 的 python package。

**注意：因應 pip 安裝的檔案大小限制，所以在第一次 import monpa 時將下載 model 檔，約 200 MB (實際大小：216681674 KB)。採分次下載，請務必等待下載完成。**

```python
import monpa
```

等看到```#已完成 monpa model 下載，歡迎使用。Download completed.```提示才表示下載完成。

如果下載不完整的 model 檔，請到 monpa package 的安裝資料夾刪除 model-511.pt 檔案，並再次執行 import monpa 來啟動下載程序。(相關討論與解法集中於 [Issue 1](https://github.com/monpa-team/monpa/issues/1))

### cut function

若只需要中文分詞結果，請用 ```cut``` function，回傳值是 list 格式。簡單範例如下：

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

若需要中文分詞及其 POS 結果，請用 ```pseg``` function，回傳值是 list of list 格式，簡單範例如下：

```python
sentence = "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。"
result = monpa.pseg(sentence)

for t in result:
  print(t)
```

輸出

```python
['蔡英文', 'PER']
['總統', 'Na']
['今天', 'Nd']
['受', 'P']
['邀', 'VF']
['參加', 'VC']
['台北市政府', 'ORG']
['所', 'D']
['舉辦', 'VC']
['的', 'DE']
['陽明山', 'LOC']
['馬拉松', 'Na']
['比賽', 'Na']
['。', 'PERIODCATEGORY']
```

### 載入自訂詞典 load_userdict function

如果需要自訂詞典，請依下列格式製作詞典文字檔，再使用此功能載入。簡單範例如下：

假設製作一個 userdict.txt 檔，每行含三部分，必須用『空格 （space）』隔開，依次是：詞語、詞頻（請填數值，目前無作用）、詞性（未能確定，請填 ```NER```），順序不可錯亂。

**注意：最後不要留空行或任何空白空間。***

```reStructuredText
台北市政府 100 NER
受邀 100 V
```

當要使用自訂詞時，請於執行分詞前先做 ```load_userdict```，將自訂詞典載入到 monpa 模組。

請將本範例的 ```./userdict.txt``` 改成實際放置自訂詞文字檔路徑及檔名。

```python
monpa.load_userdict("./userdict.txt")
```

延用前例，用 ```pseg``` function，可發現回傳值已依自訂詞典分詞，譬如『受邀』為一個詞而非先前的兩字分列輸出，『台北市政府』也依自訂詞輸出。

```python
sentence = "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。"
result = monpa.pseg(sentence)

for t in result:
  print(t)
```

輸出

```python
['蔡英文', 'PER']
['總統', 'Na']
['今天', 'Nd']
['受邀', 'V']
['參加', 'VC']
['台北市政府', 'NER']
['所', 'D']
['舉辦', 'VC']
['的', 'DE']
['陽明山', 'LOC']
['馬拉松', 'Na']
['比賽', 'Na']
['。', 'PERIODCATEGORY']
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
