# student-opinion-cooccurrence-network
---
## 緣由：
因為寫關於科技領域課課程為了生存從工藝課變到現在的生活科技的反思作業，其他非考科科目也被挑戰存在學生的堂數的用意有沒有用。剛好查到[將國高中藝能課程改成選修](https://join.gov.tw/idea/detail/7c8d9351-3f20-4ce3-9095-500633cab73d)提到藝能科浪費時間又不是每個人都有興趣提議改選修的提案，以及[國高中上課時間改為10:00到16:00，避免慢性睡眠剝奪導致學生憂鬱、自傷自殺風險提升](https://join.gov.tw/idea/detail/45e4b677-19d5-4b48-b1da-1afe3000a878)剛覆議成功，有在社群看到在討論，所以就開啟~~老本？~~（來看這裡[1201_文本共線網絡圖：台北當代藝術館與其他展館看展關聯性](https://github.com/cpeggy/PL/tree/main/Homework5)）只是用 AI 速寫 code（要教作業了拉拉拉），用共線圖討論：

當討論到**多元課程**時覆議者、討論區視角認為存廢與否？（大眾視角）

btw 發現和同學口術後覺得自己講得好心虛所以建立 Repo. 並用 REAdME 紀錄（前面討論太熱烈＆謝謝老師放假~~~）
## 步驟：
1. 把~~留言爬下來~~ 誒不用這麼累啦，有可以下載討論區和附議區的留言（.csv）
2. 過濾文本：只針對討論區留言.csv的留言內容欄位和附議名單.csv的附議原因欄位提及關於「多元」的整段文字（檔案超連結[extract_duoyuan.py](https://github.com/cpeggy/student-opinion-cooccurrence-network/blob/main/extract_duoyuan.py)）
  1. 為何用「多元」非「多元課程」？
     1.  有概括看過留言覺得多元能提到的內容比多元課成更完善（e.g. 多元發展等）
3. 使用斷詞取出現次數最高的30個字共線圖分析將這些言論中的關聯（檔案超連結[cooccurrence_analysis.py](https://github.com/cpeggy/student-opinion-cooccurrence-network/blob/main/cooccurrence_analysis.py) >> AI 太雞婆ㄌ我只需要看共線圖但他跑一堆算ㄌ）
## 結果：
### 如何解讀？
 1. 元素解釋：
  1. 節點 = 一個詞彙
    1. 節點大小 = 該詞的連結數量
    2. 越大 = 與越多詞彙共同出現
    3. 越小 = 較孤立的詞彙
  2. 線條 = 兩個詞在同一文本中出現
    1. 線條粗細 = 共現次數
    2. 越粗 = 兩詞越常一起出現
    3. 越細 = 偶爾一起出現
  3. 節點間距離近 = 關係緊密，常一起出現，節點間距離遠 = 關係較弱
### 共線圖本圖：
[!圖本圖](https://github.com/cpeggy/student-opinion-cooccurrence-network/blob/main/%E8%A9%9E%E5%BD%99%E5%85%B1%E7%8F%BE%E7%B6%B2%E7%B5%A1%E5%9C%96.png)
  1. 
