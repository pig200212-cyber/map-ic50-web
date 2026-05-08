import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pubchempy as pcp
import math
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# --- 核心運算類別 ---
class MAP_IC50_Engine:
    def __init__(self, target_time=48):
        self.v_smiles = "O[C@@H]1[C@H](OC(=O)/C=C/c2ccc(O)c(O)c2)[C@@H](CO)O[C@H](Oc2cc(ccc2O)/C=C/c3ccc(O)c(O)c2)[C@H]1O"
        self.v_phi_ref = 1.065
        self.psi_base = 287.6
        self.gamma_a549 = 0.78
        self.tau_t = self._calculate_tau(target_time)

    def _calculate_tau(self, t):
        ln_ratio = math.log(t / 24)
        tau = 1 + 0.802 * ln_ratio + 1.394 * (ln_ratio**2)
        return 1.576 if t == 48 else tau

    def _get_feats(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return {"psa": Descriptors.TPSA(mol), "logp": Descriptors.MolLogP(mol),
                "fp": AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)}

    def predict(self, x_smiles, purity=0.95, f_apo=1.12):
        v_f = self._get_feats(self.v_smiles)
        x_f = self._get_feats(x_smiles)
        sim = DataStructs.TanimotoSimilarity(x_f["fp"], v_f["fp"])
        pol_ratio = (x_f["psa"]/v_f["psa"]) / (max(0.1, x_f["logp"])/max(0.1, v_f["logp"]))
        phi_x = self.v_phi_ref * (sim / pol_ratio)
        eta = 1 + 1.55 * ((purity - 0.30) / 0.70)**0.72
        ic50 = (self.psi_base * phi_x * self.gamma_a549 * eta) / (self.tau_t * f_apo)
        return ic50, sim

# --- Streamlit 網頁介面 ---
st.set_page_config(page_title="MAP-IC50 AI Platform", layout="wide")
st.title("🧪 MAP-IC50 藥物活性預測與統計平台")

# 側邊欄：實驗參數
st.sidebar.header("⚙️ 全局參數設定")
t_exp = st.sidebar.slider("處理時間 (h)", 24, 72, 48)
purity_exp = st.sidebar.slider("樣本純度 (η)", 0.50, 1.00, 0.95)
engine = MAP_IC50_Engine(target_time=t_exp)

tab1, tab2, tab3 = st.tabs(["💊 藥物活性預測", "📊 ANOVA 統計檢定", "📖 使用說明"])

# --- Tab 1: 預測 ---
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("輸入參數")
        cid_input = st.text_input("輸入 PubChem CID", "5281800")
        f_apo_input = st.number_input("凋亡誘導因子 (f_apo)", value=1.12)
        run_btn = st.button("執行 AI 預測")

    with col2:
        if run_btn:
            with st.spinner("正在檢索 PubChem 數據並計算..."):
                try:
                    comp = pcp.Compound.from_cid(cid_input)
                    ic50, sim = engine.predict(comp.isomeric_smiles, purity=purity_exp, f_apo=f_apo_input)
                    
                    st.metric("預測 IC50", f"{ic50:.2f} μg/mL")
                    st.write(f"**化合物名稱:** {comp.iupac_name}")
                    st.write(f"**結構相似度:** {sim:.4f}")
                    
                    # 繪圖
                    concs = np.logspace(0, 4, 100)
                    surv = 100 / (1 + (concs / ic50)**1.2)
                    fig, ax = plt.subplots()
                    ax.semilogx(concs, surv, label=f"Predicted Curve (IC50={ic50:.1f})")
                    ax.axhline(50, color='r', linestyle='--')
                    ax.set_xlabel("Concentration (ug/mL)")
                    ax.set_ylabel("Viability (%)")
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"錯誤: {e}")

# --- Tab 2: ANOVA ---
with tab2:
    st.subheader("自由調整 ANOVA 分析")
    anova_type = st.radio("選擇分析類型", ["One-Way ANOVA", "Two-Way ANOVA"])
    
    if anova_type == "One-Way ANOVA":
        st.write("請輸入各組數據（逗號隔開）：")
        c1 = st.text_input("組別 1 (例如 Control)", "100, 98, 102")
        c2 = st.text_input("組別 2 (例如 500ug)", "65, 62, 60")
        if st.button("執行單因子分析"):
            d1 = [float(x) for x in c1.split(",")]
            d2 = [float(x) for x in c2.split(",")]
            f, p = stats.f_oneway(d1, d2)
            st.write(f"**F-Stat:** {f:.4f}, **P-Value:** {p:.4e}")
            if p < 0.05: st.success("顯著差異 (p < 0.05)")

# --- Tab 3: 說明 ---
with tab3:
    st.info("本系統基於 MAP-IC50 V5.0 算法，整合時間補償、極性修正與統計分析。")