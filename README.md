# 🧪 MAP-IC50 生物醫藥活性預測平台

這是一個基於 **AI 與化學資訊學** 開發的藥物篩選與統計平台。

## 🌟 核心功能
* **IC50 活性預測**：輸入化合物 CID，自動結合時間補償因子（τ）進行外推運算。
* **結構相似度分析**：利用 RDKit 計算分子指紋（Tanimoto Similarity）。
* **統計模組**：內建 One-Way 與 Two-Way ANOVA 檢定，支援三重複實驗分析。

## 🛠️ 技術棧
* **語言**: Python 3.9+
* **框架**: [Streamlit](https://streamlit.io/)
* **化學庫**: RDKit, PubChemPy
* **統計庫**: SciPy, Statsmodels

## 🚀 如何使用
1. 進入 [你的 Streamlit 網址]
2. 在側邊欄調整實驗時間（24h-72h）與樣本純度。
3. 輸入 PubChem CID 即可獲得預測曲線。

## 📜 引用與致謝
本研究模型參考 Verbascoside 實驗基準進行校正。
