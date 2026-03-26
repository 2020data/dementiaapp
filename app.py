import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from patsy import dmatrix, build_design_matrices, cr
import io

# 設定頁面標題
st.set_page_config(page_title="RCS Cox Analysis Tool", layout="wide")

st.title("📊 Restricted Cubic Spline (RCS) Cox 分析工具")
st.markdown("""
本工具可執行 Cox 比例風險模型結合限制性立方樣條 (RCS)，產出校正後的 Hazard Ratio 曲線與資料分佈圖。
""")

# --- 1. 側邊欄：檔案上傳 ---
st.sidebar.header("📁 資料導入")
uploaded_file = st.sidebar.file_uploader("上傳 Excel 檔案", type=["xlsx"])

if uploaded_file:
    # 讀取 Excel 所有欄位名稱以便選取
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)

    df_raw = load_data(uploaded_file)
    st.sidebar.success("檔案讀取成功！")

    # --- 2. 側邊欄：變數選取 ---
    st.sidebar.header("⚙️ 變數設定")
    
    # 基本生存分析變數
    time_col = st.sidebar.selectbox("追蹤時間 (Time):", df_raw.columns, index=df_raw.columns.get_loc('pointtime') if 'pointtime' in df_raw.columns else 0)
    event_col = st.sidebar.selectbox("事件標記 (Event):", df_raw.columns, index=df_raw.columns.get_loc('Conversion') if 'Conversion' in df_raw.columns else 0)
    target_var = st.sidebar.selectbox("主要分析變數 (RCS Target):", df_raw.columns, index=df_raw.columns.get_loc('LDL') if 'LDL' in df_raw.columns else 0)
    
    # 校正變數
    all_cols = df_raw.columns.tolist()
    default_cat = [v for v in ['Gender', 'HTN', 'DM', 'CAD', 'CVA', 'Anti_HTN', 'Anti_DM', 'AntiPLT', 'Antidementia'] if v in all_cols]
    default_adj = [v for v in ['Age', 'Education', 'TG', 'HDL', 'CASI', 'HAIADL', 'NPI_SB', 'CFS'] if v in all_cols]
    
    categorical_vars = st.sidebar.multiselect("類別型校正變數:", all_cols, default=default_cat)
    adjust_vars = st.sidebar.multiselect("連續型校正變數:", all_cols, default=default_adj)
    
    # 篩選條件
    filter_col = st.sidebar.selectbox("篩選欄位 (Optional):", ["None"] + all_cols)
    filter_val = None
    if filter_col != "None":
        unique_vals = df_raw[filter_col].unique().tolist()
        filter_val = st.sidebar.selectbox(f"只分析 {filter_col} 等於：", unique_vals)

    # RCS 設定
    ref_value = st.sidebar.number_input(f"設定 {target_var} 的參考點 (HR=1):", value=100.0)
    knots_df = st.sidebar.slider("樣條自由度 (df):", min_value=3, max_value=6, value=4)

    # --- 3. 主要內容區：資料處理 ---
    st.header("1. 資料預覽與清洗")
    
    # 執行篩選
    df_filtered = df_raw.copy()
    if filter_col != "None":
        df_filtered = df_filtered[df_filtered[filter_col] == filter_val]
    
    # 選取需要的欄位並移除空值
    needed_cols = [time_col, event_col, target_var] + categorical_vars + adjust_vars
    df_clean = df_filtered[needed_cols].dropna()
    
    col1, col2 = st.columns(2)
    col1.metric("原始筆數", len(df_raw))
    col2.metric("分析筆數 (清洗後)", len(df_clean))
    
    st.dataframe(df_clean.head(), use_container_width=True)

    if st.button("🚀 執行 RCS Cox 分析"):
        try:
            with st.spinner('模型計算中...'):
                # --- 4. 模型配適 ---
                # 建立公式
                formula = f"cr({target_var}, df={knots_df}) + " + " + ".join(adjust_vars) + " + " + " + ".join([f"C({v})" for v in categorical_vars])
                
                # 手動建立訓練矩陣
                design_matrix_train = dmatrix(formula, df_clean, return_type='dataframe')
                if 'Intercept' in design_matrix_train.columns:
                    design_matrix_train = design_matrix_train.drop(columns=['Intercept'])
                
                train_df = design_matrix_train.copy()
                train_df[time_col] = df_clean[time_col].values
                train_df[event_col] = df_clean[event_col].values
                
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(train_df, duration_col=time_col, event_col=event_col)
                
                # --- 5. 計算 HR 曲線 ---
                train_design_info = dmatrix(formula, df_clean).design_info
                x_vals = np.linspace(df_clean[target_var].min(), df_clean[target_var].max(), 100)
                
                pred_df = pd.DataFrame({target_var: x_vals})
                for col in adjust_vars: pred_df[col] = df_clean[col].mean()
                for col in categorical_vars: pred_df[col] = df_clean[col].mode()[0]
                
                design_matrix_pred = build_design_matrices([train_design_info], pred_df)[0]
                design_matrix_pred = pd.DataFrame(design_matrix_pred, columns=train_design_info.column_names)
                
                ref_df = pred_df.copy().iloc[:1]
                ref_df[target_var] = ref_value
                design_matrix_ref = build_design_matrices([train_design_info], ref_df)[0]
                design_matrix_ref = pd.DataFrame(design_matrix_ref, columns=train_design_info.column_names)
                
                if 'Intercept' in design_matrix_pred.columns: design_matrix_pred = design_matrix_pred.drop(columns=['Intercept'])
                if 'Intercept' in design_matrix_ref.columns: design_matrix_ref = design_matrix_ref.drop(columns=['Intercept'])
                
                model_cols = cph.params_.index
                X = design_matrix_pred[model_cols].values
                X_ref = design_matrix_ref[model_cols].values
                
                beta = cph.params_.values
                cov_matrix = cph.variance_matrix_.values
                
                delta_X = X - X_ref
                log_hr = np.dot(delta_X, beta)
                var_log_hr = np.diag(np.dot(np.dot(delta_X, cov_matrix), delta_X.T))
                se_log_hr = np.sqrt(var_log_hr)
                
                hr = np.exp(log_hr)
                upper_hr = np.exp(log_hr + 1.96 * se_log_hr)
                lower_hr = np.exp(log_hr - 1.96 * se_log_hr)

                # --- 6. 繪圖 ---
                st.header("2. 分析結果圖表")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                             gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
                
                ax1.fill_between(x_vals, lower_hr, upper_hr, color='#f7d1db', alpha=0.5, label='95% CI')
                ax1.plot(x_vals, hr, color='#a33d4e', linewidth=2.5, label='Hazard Ratio')
                ax1.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                ax1.scatter([ref_value], [1], color='black', zorder=5)
                ax1.set_ylabel('Hazard Ratio', fontsize=12)
                ax1.set_title(f'RCS Analysis: {target_var} vs {event_col}', fontsize=16)
                ax1.grid(True, linestyle=':', alpha=0.4)
                
                ax2.hist(df_clean[target_var], bins=40, color='#4b2e63', alpha=0.8, rwidth=0.9)
                ax2.set_xlabel(f'{target_var} Level', fontsize=12)
                ax2.set_ylabel('Count', fontsize=10)
                
                for ax in [ax1, ax2]:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)
                
                # --- 7. 顯示模型摘要 ---
                st.header("3. Cox 模型統計摘要")
                st.text("這部分顯示各個樣條項與校正變數的 P 值與係數：")
                st.dataframe(cph.summary, use_container_width=True)

        except Exception as e:
            st.error(f"分析出錯：{e}")
            st.info("請檢查變數是否選取正確，或資料中是否有非數值內容。")

else:
    st.info("👈 請先在側邊欄上傳 Excel 檔案以開始分析。")