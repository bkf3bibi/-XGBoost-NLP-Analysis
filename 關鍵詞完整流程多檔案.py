import jieba
import pdfplumber
import os
import glob
import pandas as pd
from collections import Counter

# ==========================================
# 1. 設定路徑
# ==========================================
folder_path = r'C:\Users\bkf3bibi\Desktop\工作練習\新報\下載檔案' # 放所有 PDF 的資料夾
dict_path = r"C:\Users\bkf3bibi\Desktop\工作練習\新報\finance_dict.txt"
industry_txt = r"C:\Users\bkf3bibi\Desktop\工作練習\新報\industry_keywords.txt.txt"

# 載入辭典
jieba.load_userdict(dict_path)

# 讀取產業動能字典
with open(industry_txt, "r", encoding="utf-8") as f:
    target_words = {line.split()[0].strip() for line in f if line.strip()}

# ==========================================
# 2. 定義單一檔案處理函數
# ==========================================
def process_pdf(file_path):
    full_text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text: full_text += text
        
        # 斷詞與過濾
        words = jieba.lcut(full_text)
        valuable_keywords = [
            w.strip() for w in words 
            if w.strip() in target_words
        ]
        return Counter(valuable_keywords)
    except Exception as e:
        print(f"處理 {os.path.basename(file_path)} 時出錯: {e}")
        return Counter()

# ==========================================
# 3. 批量執行與資料整合
# ==========================================
# 抓取資料夾內所有 PDF
pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
all_results = []

print(f"找到 {len(pdf_files)} 個 PDF 檔案，開始處理...")

for file in pdf_files:
    file_name = os.path.basename(file)
    print(f"正在分析: {file_name}...")
    
    counts = process_pdf(file)
    
    # 將統計結果轉化為長表格式 (適合後續特徵矩陣使用)
    for word, count in counts.items():
        all_results.append({
            "檔案名稱": file_name,
            "關鍵字": word,
            "出現次數": count
        })

# ==========================================
# 4. 匯出結果
# ==========================================
df = pd.DataFrame(all_results)

if not df.empty:
    # 匯出成 Excel，這就是你機器學習最棒的原始資料！
    output_path = r'C:\Users\bkf3bibi\Desktop\工作練習\新報\多檔案統計結果.xlsx'
    df.to_excel(output_path, index=False)
    print(f"\n✅ 全部處理完成！結果已儲存至: {output_path}")
    print(df.head(10)) # 預覽前10筆
else:
    print("❌ 未能統計到任何關鍵字，請檢查 PDF 內容與字典。")