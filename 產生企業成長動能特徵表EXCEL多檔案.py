import pandas as pd

# 1. 讀取「多檔案統計結果」 (假設你已經把檔案名稱拆解出「公司」與「年度」)
df = pd.read_excel('多檔案統計結果.xlsx')

# 💡 小技巧：從檔案名稱提取「公司」與「年度」 (如果原本沒有這兩欄的話)
# 假設檔名格式為: 2308_台達電_法說會簡報_202512
df['公司'] = df['檔案名稱'].str.split('_').str[1]
df['年度'] = df['檔案名稱'].str.split('_').str[-1].str[:4] # 抓出 2024 或 2025

# 2. 拆分今年與去年
df_2024 = df[df['年度'] == '2024'][['公司', '關鍵字', '出現次數']]
df_2025 = df[df['年度'] == '2025'][['公司', '關鍵字', '出現次數']]

# 3. 合併 (以 公司+關鍵字 為準)
df_compare = pd.merge(
    df_2025, df_2024, 
    on=['公司', '關鍵字'], 
    how='outer', 
    suffixes=('_今年', '_去年')
).fillna(0)

# 4. 計算動能變化率與分類
df_compare['momentum_rate'] = (df_compare['出現次數_今年'] - df_compare['出現次數_去年']) / (df_compare['出現次數_去年'] + 1)

def label_growth(row):
    if row['momentum_rate'] > 0.5: return '爆發型'
    if row['momentum_rate'] > 0: return '穩定型'
    return '衰退型'

df_compare['動能分類'] = df_compare.apply(label_growth, axis=1)

# 5. 填入 Target_Growth (這是最關鍵的一步！)
# 建立一個對照表
target_map = {
    ('台達電', '2024'): 0.3087,
    ('台達電', '2025'): 0.4653,
    ('台積電', '2024'): 0.4185,
    ('台積電', '2025'): 0.3472,
    ('光寶科', '2024'): 0.2300,
    ('光寶科', '2025'): 0.1920
}

df_compare['年度'] = '2025' 

# 這樣接下來執行這段就不會報錯了
df_compare['Target_Growth'] = df_compare.apply(
    lambda row: target_map.get((str(row['公司']), str(row['年度'])), 0), axis=1
)

# 6. 存成訓練檔
df_compare.to_excel('企業成長動能特徵表_整合版.xlsx', index=False)
print("✅ 特徵表已整合三家公司數據，準備好進行 XGBoost 訓練！")