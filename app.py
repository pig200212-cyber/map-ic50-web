import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs

# --- 核心運算類別 ---
class MAP_IC50_Engine:
    def __init__(self, target_time=48):
        # 基準：Verbascoside
        self.v_smiles = r"C[C@H]1[C@@H]([C@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H](O[C@H]2OC(=O)/C=C/C3=CC(=C(C=C3)O)O)CO)[C@@H](O)OCCN4=CC=C(C=C4)O)O)O)O"
        self.v_phi_ref = 1.065
        self.psi_base = 287.6
        self.gamma_a549 = 0.78
        self.tau_t = self._calculate_tau(target_time)

    def _calculate_tau(self, t):
        ln_ratio = math.log(t / 24)
        return 1.576 if t == 48 else (1 + 0.802 * ln_ratio + 1.394 * (ln_ratio**2))

    def _get_feats(self, smiles):
        try:
            # 清理字串中可能的隱形字元
            clean_s = "".join(smiles.split()).strip()
            mol = Chem.MolFromSmiles(clean_s)
            if not mol: return None
            return {
                "psa": Descriptors.TPSA(mol), 
                "logp": Descriptors.MolLogP(mol),
                "fp": AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            }
        except: return None

    def predict(self, x_smiles, purity=0.95, f_apo=1.12):
        v_f = self._get_feats(self.v_smiles)
        x_f = self._get_feats(x_smiles)
        if v_f is None or x_f is None: return None, None
        sim = DataStructs.TanimotoSimilarity(x_f["fp"], v_f["fp"])
        denom = (max(0.1, x_f["logp"]) / max(0.1, v_f["logp"]))
        pol_ratio = (x_f["psa"] / v_f["psa"]) / denom
        phi_x = self.v_phi_ref * (sim / max(0.01, pol_ratio))
        eta = 1 + 1.55 * ((purity - 0.30) / 0.70)**0.72
        ic50 = (self.psi_base * phi_x * self.gamma_a549 * eta) / (self.tau_t * f_apo)
        return ic50, sim

# --- 預設化合物資料庫 ---
COMPOUND_DB = {
    "Verbascoside (Acteoside)": "C[C@H]1[C@@H]([C@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H](O[C@H]2OC(=O)/C=C/C3=CC(=C(C=C3)O)O)CO)[C@@H](O)OCCN4=CC=C(C=C4)O)O)O)O",
    "Isoverbascoside": "C[C@H]1[C@@H]([C@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H](O[C@H]2O)CO[C@H]3[C@@H]([C@H]([C@@H]([C@H](O3)OCCN4=CC=C(C=C4)O)O)O)OC(=O)/C=C/C5=CC=C(C=C5)O)O)O)O)O",
    "Echinacoside": "C[C@H]1[C@@H]([C@H]([C@H]([C@H](O1)O[C@@H]2[C@H](O[C@H](O[C@H]2O[C@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)O)O)CO[C@H]4[C@@H]([C@H]([C@@H]([C@H](O4)OCCN5=CC=C(C=C5)O)O)O)OC(=O)/C=C/C6=CC=C(C=C6)O)O)O)O)O"
}

# --- Streamlit 介面 ---
st.set_page_config(page_title="MAP-IC50 AI Platform", layout="wide")
st.title("🧪 MAP-IC50 藥物活性預測與統計平台")

st.sidebar.header("⚙️ 全局參數設定")
t_exp = st.sidebar.slider("處理時間 (h)", 24, 72, 48)
purity_exp = st.sidebar.slider("樣本純度 (η)", 0.50, 1.00, 0.95)
engine = MAP_IC50_Engine(target_time=t_exp)

tab1, tab2 = st.tabs(["💊 藥物活性預測", "📊 ANOVA 統計檢定"])

with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("選擇預測對象")
        mode = st.radio("模式：", ["從資料庫選擇", "手動輸入 SMILES"])
        
        target_smiles = ""
        if mode == "從資料庫選擇":
            cpd_name = st.selectbox("選擇化合物：", list(COMPOUND_DB.keys()))
            target_smiles = COMPOUND_DB[cpd_name]
        else:
            target_smiles = st.text_area("在此貼上 SMILES：", help="請確保字串完整且無空格")
            
        f_apo_input = st.number_input("凋亡誘導因子 (f_apo)", value=1.12)
        run_btn = st.button("🚀 開始執行 AI 預測")

    with col2:
        if run_btn and target_smiles:
            with st.spinner("AI 運算中..."):
                ic50, sim = engine.predict(target_smiles, purity=purity_exp, f_apo=f_apo_input)
                if ic50 is None:
                    st.error("❌ 結構解析失敗。請檢查 SMILES 是否包含非法字元。")
                else:
                    st.success("✅ 計算完成")
                    st.metric("預測 IC50", f"{ic50:.2f} μg/mL")
                    st.write(f"**結構相似度:** {sim:.4f}")
                    
                    # 畫圖
                    concs = np.logspace(0, 4, 100)
                    surv = 100 / (1 + (concs / ic50)**1.2)
                    fig, ax = plt.subplots()
                    ax.semilogx(concs, surv, label="Predicted", color='#1f77b4', linewidth=2)
                    ax.axhline(50, color='red', linestyle='--', alpha=0.5)
                    ax.set_xlabel("Concentration (µg/mL)")
                    ax.set_ylabel("Cell Viability (%)")
                    ax.grid(True, which="both", ls="-", alpha=0.2)
                    st.pyplot(fig)
