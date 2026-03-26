import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
from patsy import dmatrix, build_design_matrices, cr
from scipy import stats
from scipy.stats import gaussian_kde
from tableone import TableOne
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
# --- 介面設定 ---
st.set_page_config(page_title="失智症數據分析系統", layout="wide")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 工具函數 (來自您的原始代碼)
# ==========================================
def get_star(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'

def add_stat_annotation(ax, x1, x2, y, h, p_val, text_prefix):
    star = get_star(p_val)
    p_text = f"p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
    full_text = f"{text_prefix}: {star} {p_text}"
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c='black')
    ax.text((x1+x2)*.5, y+h+0.1, full_text, ha='center', va='bottom', color='black', fontsize=10)

# ==========================================
# 側邊欄：導覽與檔案上傳
# ==========================================
st.sidebar.title("🧬 失智症研究分析系統")
page = st.sidebar.radio("選擇分析模組", [
    "1. 數據導入與 TableOne", 
    "2. 相關性與密度分析", 
    "3. 組間比較 (Box/Violin)", 
    "4. 存活分析 (KM Curve)",
    "5. Cox 迴歸與森林圖",
    "6. 進階 RCS 與 亞群分析"
])

uploaded_file = st.sidebar.file_uploader("上傳數據 (Excel)", type=["xlsx"])

if uploaded_file:
    @st.cache_data
    def load_data(file):
        df = pd.read_excel(file)
        # 預處理：清洗與篩選 HAI_DEM == 0
        if 'HAI_DEM' in df.columns:
            df = df[df['HAI_DEM'] == 0]
        return df.dropna(subset=['pointtime', 'Conversion', 'LDL', 'CASI'])

    df = load_data(uploaded_file)
    st.sidebar.success(f"已讀取 {len(df)} 筆樣本 (已篩選 HAI_DEM=0)")

    # 全域變數設定
    target_var = 'LDL'
    event_col = 'Conversion'
    time_col = 'pointtime'
    
    # --- 分頁邏輯 ---
    
    if page == "1. 數據導入與 TableOne":
        st.header("📊 描述性統計 (TableOne)")
        cols = st.multiselect("選擇變數", df.columns.tolist(), default=['Age', 'Gender', 'LDL', 'HDL', 'TG', 'CASI', 'Conversion'])
        cat_cols = st.multiselect("類別變數", cols, default=['Gender', 'Conversion'])
        group_by = st.selectbox("分組依據", [None] + cols, index=cols.index('Conversion')+1 if 'Conversion' in cols else 0)
        
        if st.button("生成 TableOne"):
            t1 = TableOne(df, columns=cols, categorical=cat_cols, groupby=group_by, pval=True)
            st.dataframe(t1.tableone)

    elif page == "2. 相關性與密度分析":
        st.header("📈 相關性與分佈分析")
        tab1, tab2, tab3 = st.tabs(["熱力圖", "密度散佈圖", "桑基圖"])
        
        with tab1:
            corr_cols = st.multiselect("相關矩陣變數", df.select_dtypes(include=[np.number]).columns.tolist(), default=['Age', 'LDL', 'HDL', 'TG', 'CASI'])
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
            
        with tab2:
            st.subheader("帶有邊緣直方圖的密度散佈圖")
            x_col = st.selectbox("X 軸", df.columns, index=df.columns.get_loc('CASI'))
            y_col = st.selectbox("Y 軸", df.columns, index=df.columns.get_loc('LDL'))
            
            fig = plt.figure(figsize=(10, 8))
            gs = GridSpec(4, 4, hspace=0.1, wspace=0.1)
            ax_main = fig.add_subplot(gs[1:, :3])
            ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
            ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)
            
            # 計算密度
            xy = np.vstack([df[x_col], df[y_col]])
            z = gaussian_kde(xy)(xy)
            ax_main.scatter(df[x_col], df[y_col], c=z, cmap='YlGnBu', edgecolors='k', s=30, alpha=0.8)
            sns.regplot(x=x_col, y=y_col, data=df, scatter=False, order=2, ax=ax_main, line_kws={'color':'red', 'ls':'--'})
            
            ax_top.hist(df[x_col], bins=30, color='lightsteelblue', edgecolor='black')
            ax_top.axis('off')
            ax_right.hist(df[y_col], bins=30, orientation='horizontal', color='lightsteelblue', edgecolor='black')
            ax_right.axis('off')
            st.pyplot(fig)

        with tab3:
            st.subheader("Sankey Diagram (LDL vs CASI)")
            n_groups = st.slider("分組數量 (N)", 3, 7, 5)
            df['LDL_G'] = pd.qcut(df['LDL'], q=n_groups, labels=[f'LDL_Q{i+1}' for i in range(n_groups)])
            df['CASI_G'] = pd.qcut(df['CASI'], q=n_groups, labels=[f'CASI_Q{i+1}' for i in range(n_groups)])
            flow = df.groupby(['LDL_G', 'CASI_G'], observed=True).size().reset_index(name='Value')
            all_n = list(flow['LDL_G'].unique()) + list(flow['CASI_G'].unique())
            n_dict = {n: i for i, n in enumerate(all_n)}
            fig_s = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, label=all_n, color="blue"),
                link=dict(source=flow['LDL_G'].map(n_dict), target=flow['CASI_G'].map(n_dict), value=flow['Value'])
            )])
            st.plotly_chart(fig_s)

    elif page == "3. 組間比較 (Box/Violin)":
        st.header("🧪 組間差異分析 (LDLQ4)")
        y_axis = st.selectbox("分析目標 (Y 軸)", ['CASI', 'Age', 'TG', 'HDL'])
        
        # P-value 計算
        q_groups = df.groupby('LDLQ4')[y_axis]
        data_q1 = q_groups.get_group(1).dropna()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='LDLQ4', y=y_axis, data=df, ax=ax, palette='Set2')
        sns.stripplot(x='LDLQ4', y=y_axis, data=df, ax=ax, color='black', alpha=0.3)
        
        # 加入顯著性標註 (Q1 vs Q2, Q3, Q4)
        y_max = df[y_axis].max()
        for i, q in enumerate([2, 3, 4]):
            target_data = q_groups.get_group(q).dropna()
            _, p = stats.ttest_ind(data_q1, target_data)
            add_stat_annotation(ax, 0, i+1, y_max + (i*5), 2, p, f"Q{q} vs Q1")
        
        st.pyplot(fig)

    elif page == "4. 存活分析 (KM Curve)":
        st.header("⏳ 生存分析 (Kaplan-Meier)")
        split_col = st.selectbox("分組欄位", ['Conversion', 'Gender', 'LDLQ4'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
        kmf = KaplanMeierFitter()
        naf = NelsonAalenFitter()
        
        for name, grouped_df in df.groupby(split_col):
            kmf.fit(grouped_df[time_col], grouped_df[event_col], label=f'{split_col} {name}')
            kmf.plot_survival_function(ax=ax2)
            naf.fit(grouped_df[time_col], grouped_df[event_col], label=f'{split_col} {name}')
            naf.plot_cumulative_hazard(ax=ax1)
            
        add_at_risk_counts(kmf, ax=ax2)
        ax1.set_title("Cumulative Hazard")
        ax2.set_title("Kaplan-Meier Survival")
        st.pyplot(fig)

    elif page == "5. Cox 迴歸與森林圖":
        st.header("🌲 多變項 Cox 迴歸分析")
        covs = st.multiselect("選擇自變數", df.columns.tolist(), default=['Age', 'Gender', 'LDL', 'CASI', 'HTN', 'DM'])
        
        if st.button("執行 Cox 迴歸"):
            cph = CoxPHFitter()
            cph.fit(df[covs + [time_col, event_col]], duration_col=time_col, event_col=event_col)
            
            st.write("### 模型摘要")
            st.dataframe(cph.summary[['exp(coef)', 'p', 'exp(coef) lower 95%', 'exp(coef) upper 95%']])
            
            st.write("### 森林圖 (Forest Plot)")
            fig, ax = plt.subplots(figsize=(10, 8))
            # 自訂彩色森林圖邏輯
            summary = cph.summary
            y_pos = np.arange(len(summary))
            hr = np.exp(summary['coef'])
            low = summary['exp(coef) lower 95%']
            up = summary['exp(coef) upper 95%']
            
            # --- 替換從 colors = [...] 開始到 ax.invert_yaxis() 的區塊 ---
            # 建立顏色陣列
            colors = ['#d62728' if (p < 0.05 and h > 1) else '#1f77b4' if (p < 0.05 and h <= 1) else '#7f8c8d' for p, h in zip(summary['p'], hr)]
            
            # 👇 改成用迴圈一筆一筆畫
            for i in range(len(summary)):
                ax.errorbar(
                    hr.iloc[i], 
                    y_pos[i], 
                    xerr=[[hr.iloc[i] - low.iloc[i]], [up.iloc[i] - hr.iloc[i]]], 
                    fmt='s', 
                    color=colors[i],   # 這裡每次只給單一顏色，就不會報錯了
                    ecolor=colors[i], 
                    capsize=5, 
                    markersize=10
                )
            
            ax.axvline(1, color='black', ls='--', alpha=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(summary.index)
            ax.invert_yaxis()
            for i in range(len(summary)):
                ax.errorbar(
                    hr.iloc[i], 
                    y_pos[i], 
                    xerr=[[hr.iloc[i] - low.iloc[i]], [up.iloc[i] - hr.iloc[i]]], 
                    fmt='s', 
                    color=colors[i],   # 這裡每次只給單一顏色，就不會報錯了
                    ecolor=colors[i], 
                    capsize=5, 
                    markersize=10
                )
            ax.axvline(1, color='red', ls='--')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(summary.index)
            ax.invert_yaxis()
            st.pyplot(fig)

    elif page == "6. 進階 RCS 與 亞群分析":
        st.header("📉 進階非線性與亞群分析")
        tab_rcs, tab_sub = st.tabs(["RCS 樣條圖", "亞群森林圖"])
        
        with tab_rcs:
            st.subheader("Restricted Cubic Spline (LDL vs Dementia)")
            ref_val = st.number_input("參考基準點 (Ref)", value=100.0)
            
            # 建立 RCS 模型 (簡化版邏輯)
            knots = np.nanpercentile(df['LDL'], [5, 35, 65, 95])
            formula = f"cr(LDL, knots={list(knots)})"
            dm = dmatrix(formula, df, return_type='dataframe')
            df_rcs = pd.concat([df[[time_col, event_col]], dm], axis=1).drop('Intercept', axis=1, errors='ignore')
            
            cph_rcs = CoxPHFitter(penalizer=0.1)
            cph_rcs.fit(df_rcs, duration_col=time_col, event_col=event_col)
            
            # 預測
            x_range = np.linspace(df['LDL'].min(), df['LDL'].max(), 100)
            # 使用 build_design_matrices 確保轉換一致性
            # --- 替換從 di = dmatrix(...) 開始到 hr = np.exp(log_hr) 的區塊 ---
            di = dmatrix(formula, df).design_info
            
            # 產生預測矩陣，並轉回 DataFrame
            test_dm_raw = build_design_matrices([di], pd.DataFrame({'LDL': x_range}))[0]
            ref_dm_raw = build_design_matrices([di], pd.DataFrame({'LDL': [ref_val]}))[0]
            
            test_dm = pd.DataFrame(test_dm_raw, columns=di.column_names)
            ref_dm = pd.DataFrame(ref_dm_raw, columns=di.column_names)
            
            # 【關鍵修復】：嚴格對齊訓練模型的欄位，自動過濾掉多餘的 Intercept
            model_cols = cph_rcs.params_.index
            test_dm = test_dm[model_cols]
            ref_dm = ref_dm[model_cols]
            
            # 重新轉換為 NumPy array 計算 Log HR
            beta = cph_rcs.params_.values
            log_hr = np.dot(test_dm.values - ref_dm.values, beta)
            hr = np.exp(log_hr)
            
            fig, ax = plt.subplots()
            ax.plot(x_range, hr, color='red', lw=2)
            ax.fill_between(x_range, hr*0.8, hr*1.2, alpha=0.2, color='red') # 示意 CI
            ax.axhline(1, color='black', ls='--')
            ax.set_xlabel("LDL (mg/dL)")
            ax.set_ylabel("Adjusted Hazard Ratio")
            st.pyplot(fig)
            
        with tab_sub:
            st.subheader("Subgroup Analysis")
            # 範例：年齡分群
            sub_results = []
            for label, mask in [('Age < 65', df['Age'] < 65), ('Age >= 65', df['Age'] >= 65)]:
                sub_df = df[mask]
                cph_sub = CoxPHFitter().fit(sub_df[[time_col, event_col, 'LDL']], duration_col=time_col, event_col=event_col)
                res = cph_sub.summary.loc['LDL']
                sub_results.append({'Sub': label, 'HR': np.exp(res['coef']), 'L': res['exp(coef) lower 95%'], 'U': res['exp(coef) upper 95%']})
            
            st.dataframe(pd.DataFrame(sub_results))

else:
    st.info("請上傳 Excel 檔案以開始分析。")