import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pubchempy as pcp
import math
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs

# --- 核心運算類別 ---
class MAP_IC50_Engine:
    def __init__(self, target_time=48):
        # 基準化合物：Verbascoside (5281800)
        self.v_smiles = r"O[C@@H]1[C@H](OC(=O)/C=C/c2ccc(O)c(O)c2)[C@@H](CO)O[C@H](Oc2cc(ccc2O)/C=C/c3ccc(O)c(O)c2)[C@H]1O"
        self.v_phi_ref = 1.065
        self.psi_base = 287.6
        self.gamma_a549 = 0.78
        self.tau_t = self._calculate_tau(target_time)

    def _calculate_tau(self, t):
        ln_ratio = math.log(t / 24)
        return 1.576 if t == 48 else (1 + 0.802 * ln_ratio + 1.394 * (ln_ratio**2))

    def _get_feats(self, smiles):
        if not smiles: return None
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        try:
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
        st.subheader("輸入方式選擇")
        input_method = st.radio("選擇數據來源：", ["PubChem CID 自動抓取", "手動貼上 SMILES 結構"])
        
        target_smiles = ""
        if input_method == "PubChem CID 自動抓取":
            cid_input = st.text_input("輸入 PubChem CID", "5281800")
            if st.button("從網路檢索並預測"):
                try:
                    comp = pcp.Compound.from_cid(cid_input)
                    target_smiles = getattr(comp, 'isomeric_smiles', "")
                except: st.error("無法連上 PubChem，請改用手動貼上 SMILES。")
        else:
            target_smiles = st.text_area("在此貼上化合物的 SMILES", "")
            predict_btn = st.button("執行預測")

        f_apo_input = st.number_input("凋亡誘導因子 (f_apo)", value=1.12)

    with col2:
        # 當 target_smiles 有值時觸發計算
        if target_smiles:
            with st.spinner("計算中..."):
                ic50, sim = engine.predict(target_smiles, purity=purity_exp, f_apo=f_apo_input)
                if ic50 is None:
                    st.error("結構解析失敗，請確認 SMILES 格式是否正確。")
                else:
                    st.success("計算完成！")
                    st.metric("預測 IC50", f"{ic50:.2f} μg/mL")
                    st.write(f"**相似度 (Tanimoto):** {sim:.4f}")
                    
                    concs = np.logspace(0, 4, 100)
                    surv = 100 / (1 + (concs / ic50)**1.2)
                    fig, ax = plt.subplots()
                    ax.semilogx(concs, surv, label="Predicted Curve")
                    ax.axhline(50, color='r', linestyle='--')
                    ax.set_ylabel("Viability (%)")
                    ax.set_xlabel("Conc (ug/mL)")
                    st.pyplot(fig)
