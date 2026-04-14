import pandas as pd
import xgboost as xgb
import os
import matplotlib.pyplot as plt

# 設定字體以支援中文 (若環境不支援可改回英文欄位)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 

# 1. 檔案路徑 (請確認名稱與你整合後的 Excel 一致)
file_path = r'C:\Users\bkf3bibi\Desktop\工作練習\新報\企業成長動能特徵表_整合版.xlsx'

if not os.path.exists(file_path):
    print(f"❌ 找不到檔案：{file_path}")
else:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    # --- 數據清洗 ---
    df['Target_Growth'] = pd.to_numeric(df['Target_Growth'], errors='coerce')
    df['momentum_rate'] = pd.to_numeric(df['momentum_rate'], errors='coerce')
    df['出現次數_今年'] = pd.to_numeric(df['出現次數_今年'], errors='coerce')

    # 清洗掉空值
    df = df.dropna(subset=['Target_Growth', 'momentum_rate', '出現次數_今年', '動能分類'])
    
    # 檢查目標值是否有變異性 (至少要兩種不同的成長率數字)
    if df['Target_Growth'].nunique() < 2:
        print("❌ 警告：所有數據的 Target_Growth 都相同！")
        print("這會導致模型無法學習。請確保 Excel 中包含不同公司/年份的成長率（如 0.4653 和 0.3087）。")
    else:
        print(f"📊 有效訓練樣本：{len(df)} 筆，目標值種類：{df['Target_Growth'].unique()}")

        # 2. 類別轉換 (Encoding)
        status_mapping = {'衰退型': 0, '穩定型': 1, '爆發型': 2}
        df['category_code'] = df['動能分類'].map(status_mapping)

        # 3. 定義 X 與 y
        X = df[['出現次數_今年', 'momentum_rate', 'category_code']]
        y = df['Target_Growth']

        # 4. 訓練模型
        model = xgb.XGBRegressor(
            n_estimators=100, # 稍微調高讓訓練更充分
            learning_rate=0.05, 
            max_depth=4,
            objective='reg:squarederror'
        )
        model.fit(X, y)

        # 5. 輸出重要性排名
        print("\n✅ 模型訓練成功！特徵重要性分析如下：")
        importances = model.feature_importances_
        features = X.columns
        
        # 建立結果表並排序
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)

        for _, row in imp_df.iterrows():
            print(f"📍 指標 [{row['Feature']:12}] 重要性: {row['Importance']:>8.2%}")

        # 6. 畫圖 (這張圖直接放進面試投影片！)
        plt.figure(figsize=(10, 6))
        plt.barh(imp_df['Feature'], imp_df['Importance'], color='skyblue')
        plt.xlabel('重要性得分')
        plt.title('法說會關鍵字動能 - 特徵重要性分析')
        plt.gca().invert_yaxis() # 讓重要的在上面
        plt.show()



    # 1. 讓模型進行預測
y_pred = model.predict(X)

# 2. 把預測結果合併到原本的 DataFrame 中，方便對照
df_check = df.copy()
df_check['預測成長率'] = y_pred
df_check['誤差(%)'] = (df_check['預測成長率'] - df_check['Target_Growth']).abs() * 100

# 3. 按照「誤差」排序，看看哪些預測得最準
df_check = df_check.sort_values(by='誤差(%)')

# 4. 印出前 10 筆最準的，以及最後 5 筆最不準的
print("\n🎯 【對答案結果：前 10 筆最準確的預測】")
print(df_check[['公司', '關鍵字', 'Target_Growth', '預測成長率', '誤差(%)']].head(10).to_string(index=False))

print("\n⚠️ 【對答案結果：誤差最大的 5 筆】")
print(df_check[['公司', '關鍵字', 'Target_Growth', '預測成長率', '誤差(%)']].tail(5).to_string(index=False))

# 5. 存成 Excel 讓你回家慢慢對
df_check.to_excel(r'C:\Users\bkf3bibi\Desktop\工作練習\新報\模型預測對照表.xlsx', index=False)
print(f"\n✅ 完整的對照表已存入：模型預測對照表.xlsx")