import pandas as pd
import re
from collections import Counter, defaultdict
from itertools import combinations
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 設定中文字型（根據你的系統調整）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 讀取已篩選的資料
discussion_duoyuan = pd.read_csv('討論區留言_含多元.csv')
endorsement_duoyuan = pd.read_csv('附議名單_含多元.csv')

print(f"討論區留言數量：{len(discussion_duoyuan)} 則")
print(f"附議原因數量：{len(endorsement_duoyuan)} 則")

# 定義停用詞（可根據需求調整）
stopwords = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
    '一個', '上', '也', '很', '到', '說', '要', '去', '你', '會', '著', '沒有',
    '看', '好', '自己', '這', '那', '之', '與', '及', '或', '但', '而', '等',
    '為', '以', '將', '於', '由', '讓', '把', '被', '從', '向', '更', '最',
    '能', '可', '如', '因', '所', '這樣', '那樣', '如果', '因為', '所以',
    '他', '她', '它', '我們', '你們', '他們', '什麼', '怎麼', '多少', '哪裡'
])

# 定義關鍵詞類別（用於後續分析）
keyword_categories = {
    '課程相關': ['課程', '課綱', '學科', '科目', '必修', '選修', '教育', '教學', '上課', '內容'],
    '時間壓力': ['時間', '壓力', '負擔', '睡眠', '疲累', '累', '趕', '忙', '熬夜', '休息'],
    '考試成績': ['考試', '成績', '分數', '升學', '聯考', '會考', '學測', '競爭', '排名'],
    '能力發展': ['能力', '素養', '發展', '培養', '學習', '探索', '興趣', '潛能', '技能'],
    '未來導向': ['未來', '社會', '職場', '工作', '就業', '實用', '應用', '出社會'],
    '身心健康': ['健康', '身心', '憂鬱', '自殺', '心理', '壓力', '快樂', '幸福', '身體'],
    '課程態度': ['重要', '必要', '需要', '移除', '刪除', '保留', '減少', '增加', '取消']
}

def extract_keywords(text, min_length=2):
    """提取關鍵詞"""
    if pd.isna(text):
        return []
    
    # 使用 jieba 分詞
    words = jieba.lcut(str(text))
    
    # 過濾：長度、停用詞、標點符號
    keywords = [
        word for word in words 
        if len(word) >= min_length 
        and word not in stopwords
        and not re.match(r'^[，。！？；：、\s\d]+$', word)
    ]
    
    return keywords

def calculate_cooccurrence(texts, window_size=None):
    """計算詞彙共現矩陣"""
    cooccurrence = defaultdict(lambda: defaultdict(int))
    word_counts = Counter()
    
    for text in texts:
        keywords = extract_keywords(text)
        word_counts.update(keywords)
        
        if window_size:
            # 使用滑動窗口
            for i in range(len(keywords)):
                for j in range(i+1, min(i+window_size, len(keywords))):
                    word1, word2 = sorted([keywords[i], keywords[j]])
                    cooccurrence[word1][word2] += 1
        else:
            # 計算文本內所有詞對
            for word1, word2 in combinations(set(keywords), 2):
                word1, word2 = sorted([word1, word2])
                cooccurrence[word1][word2] += 1
    
    return cooccurrence, word_counts

# ==================== 分析討論區留言 ====================
print("\n" + "=" * 80)
print("【討論區留言】多元課程用詞共線分析")
print("=" * 80)

discussion_texts = discussion_duoyuan['留言內容'].tolist()
discussion_cooccur, discussion_word_counts = calculate_cooccurrence(discussion_texts)

print("\n▼ 最常出現的詞彙 (Top 30)")
print("-" * 80)
for word, count in discussion_word_counts.most_common(30):
    print(f"{word:12s} : {count:4d} 次")

print("\n▼ 與「多元」共現最多的詞彙 (Top 20)")
print("-" * 80)
duoyuan_cooccur_discussion = []
for word1 in discussion_cooccur:
    if word1 == '多元':
        for word2, count in discussion_cooccur[word1].items():
            duoyuan_cooccur_discussion.append((word2, count))
    elif '多元' in discussion_cooccur[word1]:
        duoyuan_cooccur_discussion.append((word1, discussion_cooccur[word1]['多元']))

duoyuan_cooccur_discussion.sort(key=lambda x: x[1], reverse=True)
for word, count in duoyuan_cooccur_discussion[:20]:
    print(f"多元 + {word:12s} : {count:4d} 次")

# ==================== 分析附議原因 ====================
print("\n" + "=" * 80)
print("【附議原因】多元課程用詞共線分析")
print("=" * 80)

endorsement_texts = endorsement_duoyuan['附議原因'].dropna().tolist()
print(f"有效附議原因數量：{len(endorsement_texts)}")

if len(endorsement_texts) > 0:
    endorsement_cooccur, endorsement_word_counts = calculate_cooccurrence(endorsement_texts)
    
    print("\n▼ 最常出現的詞彙 (Top 30)")
    print("-" * 80)
    for word, count in endorsement_word_counts.most_common(30):
        print(f"{word:12s} : {count:4d} 次")
    
    print("\n▼ 與「多元」共現最多的詞彙 (Top 20)")
    print("-" * 80)
    duoyuan_cooccur_endorsement = []
    for word1 in endorsement_cooccur:
        if word1 == '多元':
            for word2, count in endorsement_cooccur[word1].items():
                duoyuan_cooccur_endorsement.append((word2, count))
        elif '多元' in endorsement_cooccur[word1]:
            duoyuan_cooccur_endorsement.append((word1, endorsement_cooccur[word1]['多元']))
    
    duoyuan_cooccur_endorsement.sort(key=lambda x: x[1], reverse=True)
    for word, count in duoyuan_cooccur_endorsement[:20]:
        print(f"多元 + {word:12s} : {count:4d} 次")
else:
    print("附議原因中較少提及「多元」，無法進行分析")

# ==================== 合併分析 ====================
print("\n" + "=" * 80)
print("【合併分析】討論區留言 + 附議原因")
print("=" * 80)

all_texts = discussion_texts + endorsement_texts
all_cooccur, all_word_counts = calculate_cooccurrence(all_texts)

print("\n▼ 整體最常出現的詞彙 (Top 30)")
print("-" * 80)
for word, count in all_word_counts.most_common(30):
    print(f"{word:12s} : {count:4d} 次")

print("\n▼ 整體與「多元」共現最多的詞彙 (Top 20)")
print("-" * 80)
duoyuan_cooccur_all = []
for word1 in all_cooccur:
    if word1 == '多元':
        for word2, count in all_cooccur[word1].items():
            duoyuan_cooccur_all.append((word2, count))
    elif '多元' in all_cooccur[word1]:
        duoyuan_cooccur_all.append((word1, all_cooccur[word1]['多元']))

duoyuan_cooccur_all.sort(key=lambda x: x[1], reverse=True)
for word, count in duoyuan_cooccur_all[:20]:
    print(f"多元 + {word:12s} : {count:4d} 次")

print("\n▼ 最常共現的詞對組合 (Top 20，排除多元)")
print("-" * 80)
all_pairs = []
for word1 in all_cooccur:
    for word2, count in all_cooccur[word1].items():
        if '多元' not in [word1, word2]:
            all_pairs.append((word1, word2, count))

all_pairs.sort(key=lambda x: x[2], reverse=True)
for word1, word2, count in all_pairs[:20]:
    print(f"{word1:10s} + {word2:12s} : {count:4d} 次")

# ==================== 詞彙類別分析 ====================
print("\n" + "=" * 80)
print("【詞彙類別分佈分析】")
print("=" * 80)

category_counts = defaultdict(int)
for text in all_texts:
    keywords = extract_keywords(text)
    for keyword in keywords:
        for category, words in keyword_categories.items():
            if keyword in words:
                category_counts[category] += 1

print("\n各類別詞彙出現次數：")
for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{category:12s} : {count:4d} 次")

# ==================== 立場差異分析 ====================
print("\n" + "=" * 80)
print("【不同立場的用詞特徵】（僅分析討論區留言）")
print("=" * 80)

def simple_stance(text):
    if pd.isna(text):
        return 'unknown'
    text_lower = str(text).lower()
    support_words = ['保留', '重要', '需要', '培養', '不應該移除', '不該移除', '有助於']
    oppose_words = ['移除', '刪除', '取消', '廢除', '減少多元', '不需要多元', '浪費']
    
    support_score = sum(1 for w in support_words if w in text_lower)
    oppose_score = sum(1 for w in oppose_words if w in text_lower)
    
    if support_score > oppose_score:
        return 'support'
    elif oppose_score > support_score:
        return 'oppose'
    else:
        return 'neutral'

discussion_duoyuan['stance'] = discussion_duoyuan['留言內容'].apply(simple_stance)

print(f"\n立場分佈：")
print(discussion_duoyuan['stance'].value_counts())

for stance_name, stance_label in [('支持保留多元', 'support'), ('支持移除多元', 'oppose')]:
    stance_texts = discussion_duoyuan[discussion_duoyuan['stance'] == stance_label]['留言內容'].tolist()
    if len(stance_texts) > 0:
        stance_words = []
        for text in stance_texts:
            stance_words.extend(extract_keywords(text))
        
        print(f"\n▼ {stance_name}者常用詞彙 (Top 15)")
        print("-" * 80)
        stance_counter = Counter(stance_words)
        for word, count in stance_counter.most_common(15):
            print(f"{word:12s} : {count:4d} 次")

# ==================== 儲存結果 ====================
print("\n" + "=" * 80)
print("【儲存分析結果】")
print("=" * 80)

# 轉換共現矩陣為 DataFrame
cooccur_list = []
for word1 in all_cooccur:
    for word2, count in all_cooccur[word1].items():
        cooccur_list.append({
            '詞彙1': word1,
            '詞彙2': word2,
            '共現次數': count
        })

cooccur_df = pd.DataFrame(cooccur_list).sort_values('共現次數', ascending=False)
cooccur_df.to_csv('詞彙共現分析_合併.csv', index=False, encoding='utf-8-sig')

# 儲存詞頻統計
word_freq_df = pd.DataFrame(all_word_counts.most_common(), columns=['詞彙', '出現次數'])
word_freq_df.to_csv('詞頻統計_合併.csv', index=False, encoding='utf-8-sig')

# 分別儲存討論區和附議的分析
discussion_cooccur_list = []
for word1 in discussion_cooccur:
    for word2, count in discussion_cooccur[word1].items():
        discussion_cooccur_list.append({
            '詞彙1': word1,
            '詞彙2': word2,
            '共現次數': count
        })

pd.DataFrame(discussion_cooccur_list).sort_values('共現次數', ascending=False).to_csv(
    '詞彙共現分析_討論區.csv', index=False, encoding='utf-8-sig'
)

if len(endorsement_texts) > 0:
    endorsement_cooccur_list = []
    for word1 in endorsement_cooccur:
        for word2, count in endorsement_cooccur[word1].items():
            endorsement_cooccur_list.append({
                '詞彙1': word1,
                '詞彙2': word2,
                '共現次數': count
            })
    
    pd.DataFrame(endorsement_cooccur_list).sort_values('共現次數', ascending=False).to_csv(
        '詞彙共現分析_附議.csv', index=False, encoding='utf-8-sig'
    )

print("\n結果已儲存至：")
print("- 詞彙共現分析_合併.csv（討論區+附議的合併分析）")
print("- 詞頻統計_合併.csv（合併的詞頻統計）")
print("- 詞彙共現分析_討論區.csv（僅討論區留言）")
if len(endorsement_texts) > 0:
    print("- 詞彙共現分析_附議.csv（僅附議原因）")

# ==================== 視覺化 ====================
print("\n" + "=" * 80)
print("【生成視覺化圖表】")
print("=" * 80)

# 1. 詞頻長條圖
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

top_words = all_word_counts.most_common(20)
words, counts = zip(*top_words)

ax1.barh(range(len(words)), counts, color='steelblue')
ax1.set_yticks(range(len(words)))
ax1.set_yticklabels(words)
ax1.invert_yaxis()
ax1.set_xlabel('出現次數')
ax1.set_title('最常出現的詞彙 Top 20', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# 2. 與「多元」共現的詞彙
duoyuan_top = duoyuan_cooccur_all[:20]
if duoyuan_top:
    duo_words, duo_counts = zip(*duoyuan_top)
    
    ax2.barh(range(len(duo_words)), duo_counts, color='coral')
    ax2.set_yticks(range(len(duo_words)))
    ax2.set_yticklabels(duo_words)
    ax2.invert_yaxis()
    ax2.set_xlabel('共現次數')
    ax2.set_title('與「多元」共現最多的詞彙 Top 20', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('詞頻分析圖.png', dpi=300, bbox_inches='tight')
print("✓ 已生成：詞頻分析圖.png")

# 3. 詞彙類別圓餅圖
if category_counts:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    categories = list(category_counts.keys())
    values = list(category_counts.values())
    colors = plt.cm.Set3(range(len(categories)))
    
    wedges, texts, autotexts = ax.pie(values, labels=categories, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 11})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('詞彙類別分佈', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('詞彙類別分佈圖.png', dpi=300, bbox_inches='tight')
    print("✓ 已生成：詞彙類別分佈圖.png")

# 4. 立場用詞比較圖
support_texts = discussion_duoyuan[discussion_duoyuan['stance'] == 'support']['留言內容'].tolist()
oppose_texts = discussion_duoyuan[discussion_duoyuan['stance'] == 'oppose']['留言內容'].tolist()

if len(support_texts) > 0 and len(oppose_texts) > 0:
    support_words = []
    oppose_words = []
    
    for text in support_texts:
        support_words.extend(extract_keywords(text))
    for text in oppose_texts:
        oppose_words.extend(extract_keywords(text))
    
    support_counter = Counter(support_words)
    oppose_counter = Counter(oppose_words)
    
    support_top = dict(support_counter.most_common(15))
    oppose_top = dict(oppose_counter.most_common(15))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 支持保留
    words_s = list(support_top.keys())
    counts_s = list(support_top.values())
    ax1.barh(range(len(words_s)), counts_s, color='green', alpha=0.7)
    ax1.set_yticks(range(len(words_s)))
    ax1.set_yticklabels(words_s)
    ax1.invert_yaxis()
    ax1.set_xlabel('出現次數')
    ax1.set_title('支持保留多元者常用詞彙 Top 15', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 支持移除
    words_o = list(oppose_top.keys())
    counts_o = list(oppose_top.values())
    ax2.barh(range(len(words_o)), counts_o, color='red', alpha=0.7)
    ax2.set_yticks(range(len(words_o)))
    ax2.set_yticklabels(words_o)
    ax2.invert_yaxis()
    ax2.set_xlabel('出現次數')
    ax2.set_title('支持移除多元者常用詞彙 Top 15', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('立場用詞比較圖.png', dpi=300, bbox_inches='tight')
    print("✓ 已生成：立場用詞比較圖.png")

# 5. 共現網絡圖（Top 30 詞對）
try:
    import networkx as nx
    
    G = nx.Graph()
    
    # 取最強的30個共現關係
    top_pairs = all_pairs[:50]
    
    for word1, word2, count in top_pairs:
        G.add_edge(word1, word2, weight=count)
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 繪製邊
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights)
    
    nx.draw_networkx_edges(G, pos, width=[w/max_weight*5 for w in weights], 
                          alpha=0.3, edge_color='gray')
    
    # 繪製節點
    node_sizes = [G.degree(node) * 300 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.8, 
                          edgecolors='steelblue', linewidths=2)
    
    # 繪製標籤
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold',
                           font_family='sans-serif')
    
    ax.set_title('詞彙共現網絡圖 (Top 50 詞對)', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('詞彙共現網絡圖50.png', dpi=300, bbox_inches='tight')
    print("✓ 已生成：詞彙共現網絡圖.png")
    
except ImportError:
    print("⚠ 未安裝 networkx，跳過網絡圖生成")
    print("  提示：執行 pip install networkx 可啟用網絡圖功能")

# 6. 熱力圖（高頻詞共現矩陣）
top_30_words = [word for word, count in all_word_counts.most_common(30)]
matrix_size = min(20, len(top_30_words))
top_words_for_matrix = top_30_words[:matrix_size]

# 建立共現矩陣
cooccur_matrix = np.zeros((matrix_size, matrix_size))
for i, word1 in enumerate(top_words_for_matrix):
    for j, word2 in enumerate(top_words_for_matrix):
        if i < j:
            if word1 in all_cooccur and word2 in all_cooccur[word1]:
                cooccur_matrix[i][j] = all_cooccur[word1][word2]
                cooccur_matrix[j][i] = all_cooccur[word1][word2]
            elif word2 in all_cooccur and word1 in all_cooccur[word2]:
                cooccur_matrix[i][j] = all_cooccur[word2][word1]
                cooccur_matrix[j][i] = all_cooccur[word2][word1]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cooccur_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
            xticklabels=top_words_for_matrix, yticklabels=top_words_for_matrix,
            cbar_kws={'label': '共現次數'}, ax=ax)
ax.set_title(f'高頻詞彙共現熱力圖 (Top {matrix_size})', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('詞彙共現熱力圖.png', dpi=300, bbox_inches='tight')
print("✓ 已生成：詞彙共現熱力圖.png")

print("\n所有圖表已生成完成！")
print("\n分析完成！")