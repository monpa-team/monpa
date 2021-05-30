# 罔拍 MONPA: Multi-Objective NER POS Annotator

MONPA 罔拍是一個提供正體中文斷詞、詞性標註以及命名實體辨識的多任務模型。初期只有網站示範版本（<http://nlp.tmu.edu.tw:8080/chunk>），本計劃是將 monpa 包裝成可以 pip install 的 python package (最新版本 v0.3.2)。

最新版的 monpa model 是使用 pytorch 1.x 框架訓練出來的模型，所以在使用本版本前，請先安裝 torch 1.* 以上版本才能正常使用 monpa 套件。

## 公告
```diff
- 本次更新版本 v0.3.2：解決 issue 10, 11 的建議，新增 short_sentence 斷句功能, cut_mp 及 cut_pseg 多執行程序功能等輔助程式。
- 前一版本 v0.3.1：新增運用 GPU 的批次斷詞功能 cut_batch 及 pseg_batch。
- 前版本 v0.3.0：更小，更快，依然準確。完成 pip install 後不需要再另行下載模型檔。
- 公開釋出的 MONPA 僅供學術使用，請勿使用於商業用途。本團隊亦提供針對專業領域客製模型之服務，歡迎聯絡我們。
```

MONPA v0.2+ 版本是基於 BERT（雙向 Transformer）[[1]](#1)模型來取得更強健的詞向量（word embeddings）並配合 CRF 同時進行斷詞、詞性標註、及 NER 等多個目標。已與 MONPA v0.1 版本有相當大差異，訓練語料亦與論文內容不同。

MONPA v0.3+ 版本基於 ALBERT [[2]](#2) 重新訓練，大幅降低模型檔的大小，並加快執行效率。

<a id="1">[1]</a>  BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, NAACL-HLT 2019.

<a id="2">[2]</a>  ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.
Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut, ICLR 2020.

**開發正體中文自然語言斷詞套件是一個基礎，接續的研究需要多方支持，歡迎[您的捐款](http://nlp.tmu.edu.tw/Donate/index.html)。**

monpa 各版本的斷詞效率比較圖

<img src="./monpa_2vs3.png" style="zoom:24%;" />

以上於 Google Colab 環境測試(monpa.cut 皆使用 CPU，monpa.cut_batch 使用 GPU)


**注意**

1. 建議以原文輸入 monpa 完成斷詞後，再視需求濾掉停留字（stopword）及標點符號（punctuation）。
2. 每次輸入到 monpa 做斷詞的原文超過 200 字元的部分將被截斷丟失，建議先完成合適長度分句後再應用 monpa 斷詞。可參考 wiki [如何將長文切成短句再用 monpa 斷詞？](https://github.com/monpa-team/monpa/wiki/Example-1：將長句處理成短句再運用-monpa-完成分詞)）自行開發或是使用 v0.3.2 （含）之後版本的功能程式 short_sentence 來協助分句。
3. 支援 python >= 3.6，不支援 python 2.x。

## 安裝 monpa 套件

monpa 已經支援使用 pip 指令安裝，各作業系統的安裝步驟都相同。

```bash
pip install monpa
```

安裝時將自動檢查有無 torch >= 1.0 及 requests 等套件，若無則由 pip 直接安裝。Windows 作業系統需手動安裝，建議移駕 [pytorch.org](https://www.pytorch.org) 取得最適合作業系統版本的安裝指令。

*若已經安裝 monpa v0.2.x 版本，可以``` pip install --upgrade monpa```直接升級或是先以```pip uninstall monpa``` 指令移除舊版本再行安裝新版本。*

## 使用 monpa 的簡單範例

引入 monpa 的 python package。

```python
import monpa
```

### cut function

若只需要中文斷詞結果，請用 ```cut``` function，回傳值是 list 格式。簡單範例如下：

```python
sentence = "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。"
result_cut = monpa.cut(sentence)

for item in result_cut:
    print(item)
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

若需要中文斷詞及其 POS 結果，請用 ```pseg``` function，回傳值是 list of tuples 格式，簡單範例如下：

```python
sentence = "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。"
result_pseg = monpa.pseg(sentence)

for item in result_pseg:
    print(item)
```

輸出

```python
('蔡英文', 'PER')
('總統', 'Na')
('今天', 'Nd')
('受', 'P')
('邀', 'VF')
('參加', 'VC')
('台北市政府', 'ORG')
('所', 'D')
('舉辦', 'VC')
('的', 'DE')
('陽明山', 'LOC')
('馬拉松', 'Na')
('比賽', 'Na')
('。', 'PERIODCATEGORY')
```

### load_userdict function

如果需要自訂詞典，請依下列格式製作詞典文字檔，再使用此功能載入。簡單範例如下：

假設製作一個 userdict.txt 檔，每行含三部分，必須用「空格 （space）」隔開，依次是：詞語、詞頻（數值型態）、詞性（未能確定，請填 ```NER```）。排序是以詞頻數值大者優先，若詞頻數值相同則排列前面者優先。

**注意：最後不要留空行或任何空白空間。***

```reStructuredText
台北市政府 100 NER
受邀 100 V
```

當要使用自訂詞時，請於執行斷詞前先做 ```load_userdict```，將自訂詞典載入到 monpa 模組。

請將本範例的 ```./userdict.txt``` 改成實際放置自訂詞文字檔路徑及檔名。

```python
monpa.load_userdict("./userdict.txt")
```

延用前例，用 ```pseg``` function，可發現回傳值已依自訂詞典斷詞，譬如「受邀」為一個詞而非先前的兩字分列輸出，「台北市政府」也依自訂詞輸出。

```python
sentence = "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。"
result_pseg_userdict = monpa.pseg(sentence)

for item in result_pseg_userdict:
    print(item)
```

輸出

```python
('蔡英文', 'PER')
('總統', 'Na')
('今天', 'Nd')
('受邀', 'V')
('參加', 'VC')
('台北市政府', 'NER')
('所', 'D')
('舉辦', 'VC')
('的', 'DE')
('陽明山', 'LOC')
('馬拉松', 'Na')
('比賽', 'Na')
('。', 'PERIODCATEGORY')
```
### cut_batch function

開始批次斷句前，請先啟動使用 GPU 之設定。

```python
monpa.use_gpu(True)
```

從 monpa v0.3.1 開始提供應用 GPU 運算能力的 ```cut_batch``` function，輸入須為 list 格式，單批次的輸入量需考量 GPU 的記憶體容量，回傳值亦是 list 格式。初次啟動需耗費較多時間，建議若非大量斷詞，可使用 ```cut``` function 即可。簡單範例如下：

```python
sentence_list = ["蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。", "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。"]
result_cut_batch = monpa.cut_batch(sentence_list)

for item in result_cut_batch:
    print(item)
```

輸出

```python
['蔡英文', '總統', '今天', '受', '邀', '參加', '台北市政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。']
['蔡英文', '總統', '今天', '受', '邀', '參加', '台北市政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。']
```
教學文 [使用 monpa 以 GPU 做批次斷詞](https://wyeh.medium.com/howto-monpa-cut-batch-d1fe071cc4ac?source=friends_link&sk=ee66796916038139d02e6e828c0f39a7)

### pseg_batch function

開始批次斷句前，請先啟動使用 GPU 之設定。

```python
monpa.use_gpu(True)
```

從 monpa v0.3.1 開始提供應用 GPU 運算能力的 ```pseg_batch``` function，輸入須為 list 格式，單批次的輸入量需考量 GPU 的記憶體容量，回傳值亦是 list of turples 格式。初次啟動需耗費較多時間，建議若非大量斷詞，可使用 ```pseg``` function 即可。簡單範例如下：

```python
sentence_list = ["蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。", "蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。"]
result_pseg_batch = monpa.pseg_batch(sentence_list)

for item in result_pseg_batch:
    print(item)
```

輸出

```python
[('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'P'), ('邀', 'VF'), ('參加', 'VC'), ('台北市政府', 'ORG'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY')]
[('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'P'), ('邀', 'VF'), ('參加', 'VC'), ('台北市政府', 'ORG'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY')]
```

## 輔助功能程式（ v0.3.2 開始提供）
### utils.short_sentence function

開始使用輔助功能程式前，請先載入 monpa 附屬之 utils 功能。

```python
from monpa import utils
```

基於 monpa 斷詞只處理 200 字元內的短句，所以建議先將長句分成多個短句再做斷詞才不會因過長語句而丟失斷詞。從 monpa v0.3.2 開始提供以 "。"，"！"，"？"，"，" 依序為參考斷點的 ```short_sentence``` function，輸入須為 string 格式，回傳值是 list 格式。該功能程式將先尋找 200 字元內最後一個 "。" 為斷點，若無，則改以 "！" 為斷點，以此類推。若 200 字元內皆無法找到預設 4 個標點符號為斷點來分句，就直接從 200 字元處分句。簡單範例如下：

```python
long_sentence = '''
蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。
'''
sentence_list = utils.short_sentence(long_sentence)
for item in sentence_list:
    print(item)
```

輸出

可以發現有 292 字元的 ```long_sentence``` 長句，經 ```utils.short_sentence``` 以 "。" 為斷點分成兩個短句。
```python
蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。
蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。
```
教學文 [截長句成短句，再用 monpa 來斷詞。](https://wyeh.medium.com/howto-split-long-sentence-and-use-monpa-c1221146c1b6?source=friends_link&sk=0f39a76adee36bbc0413924de123d2a1)

### utils.cut_mp function

從 monpa v0.3.1 開始提供應用 GPU 運算能力的 ```cut_batch``` function，但考量不是每台機器皆有 GPU，所以從 v0.3.2 開始提供多執行程序的功能程式來降低多量句子的斷詞耗時。輸入為 list 或是 list of list 格式，再依機器的 CPU 內核配備指定同時啟動的 worker 數量，回傳值是 list  或是 list of list 格式。初次啟動需耗費較多時間，建議若非大量斷詞，可使用 ```cut``` function 即可。簡單範例如下：

```python
sentence_list = ['蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。', '蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。']

result_cut_mp = utils.cut_mp(sentence_list, 4) #本例是指定啟動 4 個 workers
print(result_cut_mp)
```

輸出

```python
[['蔡英文', '總統', '今天', '受', '邀', '參加', '台北市政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。', '蔡英文', '總統', '今天', '受', '邀', '參加', '台北市', '政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。', '蔡英文', '總統', '今天', '受', '邀', '參加', '台北市', '政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。', '蔡英文', '總統', '今天', '受', '邀', '參加', '台北市', '政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。', '蔡英文', '總統', '今天', '受邀', '參加', '台北市', '政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。', '蔡英文', '總統', '今天', '受', '邀', '參加', '台北市', '政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。'], ['蔡英文', '總統', '今天', '受', '邀', '參加', '台北市政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。', '蔡英文', '總統', '今天', '受', '邀', '參加', '台北市政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。', '蔡英文', '總統', '今天', '受', '邀', '參加', '台北市', '政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。', '蔡英文', '總統', '今天', '受', '邀', '參加', '台北市', '政府', '所', '舉辦', '的', '陽明山', '馬拉松', '比賽', '。']]
```
教學文 [使用 monpa 多工程序加快斷詞](https://wyeh.medium.com/howto-monpa-cut-mp-74357b62bf8d?source=friends_link&sk=888b50b9d2846505c76b80e1539a55f4)

### utils.pseg_mp function

從 monpa v0.3.1 開始提供應用 GPU 運算能力的 ```cut_batch``` function，但考量不是每台機器皆有 GPU，所以從 v0.3.2 開始提供多執行程序的功能程式來降低多量句子的斷詞耗時。輸入為 list 或是 list of list 格式，再依機器的 CPU 內核配備指定同時啟動的 worker 數量，回傳值是 list  或是 list of list 格式。初次啟動需耗費較多時間，建議若非大量斷詞，可使用 ```pseg``` function 即可。簡單範例如下：

```python
sentence_list = ['蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。', '蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。蔡英文總統今天受邀參加台北市政府所舉辦的陽明山馬拉松比賽。']

result_pseg_mp = utils.pseg_mp(sentence_list, 4) #本例是指定啟動 4 個 workers
print(result_pseg_mp)
```

輸出

```python
[[('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'P'), ('邀', 'VF'), ('參加', 'VC'), ('台北市政府', 'ORG'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY'), ('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'VJ'), ('邀', 'VF'), ('參加', 'VC'), ('台北市', 'LOC'), ('政府', 'Na'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY'), ('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'VJ'), ('邀', 'VF'), ('參加', 'VC'), ('台北市', 'LOC'), ('政府', 'Na'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY'), ('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'VJ'), ('邀', 'VF'), ('參加', 'VC'), ('台北市', 'LOC'), ('政府', 'Nc'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY'), ('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受邀', 'VJ'), ('參加', 'VC'), ('台北市', 'LOC'), ('政府', 'Nc'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY'), ('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'VJ'), ('邀', 'VF'), ('參加', 'VC'), ('台北市', 'LOC'), ('政府', 'Na'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY')], [('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'P'), ('邀', 'VF'), ('參加', 'VC'), ('台北市政府', 'ORG'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY'), ('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'P'), ('邀', 'VF'), ('參加', 'VC'), ('台北市政府', 'ORG'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY'), ('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'VJ'), ('邀', 'VF'), ('參加', 'VC'), ('台北市', 'LOC'), ('政府', 'Na'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY'), ('蔡英文', 'PER'), ('總統', 'Na'), ('今天', 'Nd'), ('受', 'VJ'), ('邀', 'VF'), ('參加', 'VC'), ('台北市', 'LOC'), ('政府', 'Nc'), ('所', 'D'), ('舉辦', 'VC'), ('的', 'DE'), ('陽明山', 'LOC'), ('馬拉松', 'Na'), ('比賽', 'Na'), ('。', 'PERIODCATEGORY')]]
```

## 捐款

我們需要您的支持來延續開發自然語言的基礎設施程式，懇請捐款[臺北醫學大學自然語言處理實驗室『人工智慧卓越創新計畫』。](http://nlp.tmu.edu.tw/Donate/index.html)

## 其他

This project is inspired by our paper [MONPA: Multi-objective Named-entity and Part-of-speech Annotator for Chinese using Recurrent Neural Network](https://www.aclweb.org/anthology/papers/I/I17/I17-2014/) in which more information about the model detail can be found. 

For your reference, although we list the paper here, it does NOT mean we use the exact same corpora when training the released model. The current MONPA is a new development by adopting the (AL)BERT model and a new paper will be published later. In the meantime, we list the original paper about the core ideas of MONPA for citation purposes.

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

##### Contact
Please feel free to contact monpa team by email.
monpa.cut@gmail.com

## 致謝

茲因模型開發初期使用中央研究院中文詞知識庫小組開發之 CKIP 程式進行部分語料標註工作，後再經其他程序完成標註校正，感謝中央研究院中文詞知識庫小組的協助。MONPA 於經中央研究院中文詞知識庫小組同意下，使用 CKIP 斷詞元件輔助製作初期訓練資料。

Ma, Wei-Yun and Keh-Jiann Chen, 2003, "Introduction to CKIP Chinese Word Segmentation System for the First International Chinese Word Segmentation Bakeoff", Proceedings of ACL, Second SIGHAN Workshop on Chinese Language Processing, pp168-171.。

## License

[![CC BY-NC-SA 4.0](https://camo.githubusercontent.com/6887feb0136db5156c4f4146e3dd2681d06d9c75/68747470733a2f2f692e6372656174697665636f6d6d6f6e732e6f72672f6c2f62792d6e632d73612f342e302f38387833312e706e67)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

Copyright (c) 2020 The MONPA team under the [CC-BY-NC-SA 4.0 License](http://creativecommons.org/licenses/by-nc-sa/4.0/). All rights reserved.

僅供學術使用，請勿使用於營利目的。若您需要應用 MONPA 於商業用途，請聯繫我們協助後續事宜。（monpa.cut@gmail.com）
