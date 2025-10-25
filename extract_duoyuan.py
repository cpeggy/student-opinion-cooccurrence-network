import pandas as pd

# 讀取 CSV 檔案
discussion_df = pd.read_csv('討論區留言.csv')
endorsement_df = pd.read_csv('附議名單.csv')

print("=" * 80)
print("討論區留言中提及「多元」的內容")
print("=" * 80)

# 從討論區留言中提取包含「多元」的留言內容
discussion_with_duoyuan = discussion_df[
    discussion_df['留言內容'].str.contains('多元', na=False)
]

for idx, row in discussion_with_duoyuan.iterrows():
    print(f"\n【留言 {idx + 1}】")
    print(f"留言者：{row['留言者名稱']}")
    print(f"時間：{row['留言時間']}")
    print(f"內容：{row['留言內容']}")
    print("-" * 80)

print(f"\n共找到 {len(discussion_with_duoyuan)} 則包含「多元」的留言\n")

print("=" * 80)
print("附議名單中提及「多元」的附議原因")
print("=" * 80)

# 從附議名單中提取包含「多元」的附議原因
endorsement_with_duoyuan = endorsement_df[
    endorsement_df['附議原因'].str.contains('多元', na=False)
]

for idx, row in endorsement_with_duoyuan.iterrows():
    print(f"\n【附議 {idx + 1}】")
    print(f"附議人：{row['附議人暱稱']}")
    print(f"時間：{row['附議時間']}")
    print(f"原因：{row['附議原因']}")
    print("-" * 80)

print(f"\n共找到 {len(endorsement_with_duoyuan)} 則包含「多元」的附議原因\n")

# 將結果儲存到新的 CSV 檔案（選用）
discussion_with_duoyuan.to_csv('討論區留言_含多元.csv', index=False, encoding='utf-8-sig')
endorsement_with_duoyuan.to_csv('附議名單_含多元.csv', index=False, encoding='utf-8-sig')

print("結果已儲存至：")
print("- 討論區留言_含多元.csv")
print("- 附議名單_含多元.csv")
