import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- 讀取 summary 資料夾中的所有 txt 報告 ---
summary_dir = 'summary'
model_reports = {}

for filename in os.listdir(summary_dir):
    if filename.endswith('.txt'):
        model_name = filename.replace('.txt', '')
        with open(os.path.join(summary_dir, filename), 'r', encoding='utf-8') as f:
            report_text = f.read()

        # 使用正則擷取報告裡的 macro avg、weighted avg、accuracy
        match_macro = re.search(r'macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_text)
        match_weighted = re.search(r'weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', report_text)
        match_acc = re.search(r'accuracy\s+([\d.]+)', report_text)

        if match_macro and match_weighted and match_acc:
            macro_precision, macro_recall, macro_f1 = map(float, match_macro.groups())
            weighted_precision, weighted_recall, weighted_f1 = map(float, match_weighted.groups())
            accuracy = float(match_acc.group(1))

            model_reports[model_name] = {
                'Accuracy': accuracy,
                'Macro F1': macro_f1,
                'Macro Precision': macro_precision,
                'Macro Recall': macro_recall,
                'Weighted F1': weighted_f1,
                'Weighted Precision': weighted_precision,
                'Weighted Recall': weighted_recall,
            }

# --- 轉換為 DataFrame ---
df = pd.DataFrame.from_dict(model_reports, orient='index')
df = df.sort_values(by='Macro F1', ascending=False)

# --- Bar Chart: Macro F1, Accuracy ---
plt.figure(figsize=(12, 6))
df[['Macro F1', 'Accuracy']].plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison (Macro F1 & Accuracy)')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('summary/model_performance_comparison.png')

plt.clf()

df_macro = df[["Macro Precision", "Macro Recall", "Macro F1"]]

# heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_macro, annot=True, fmt=".2f", cmap="YlOrRd", cbar=True)

plt.title("Model Performance Heatmap (Macro Metrics)")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("summary/model_performance_heatmap_macro.png")
