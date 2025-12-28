import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
from scipy import stats
import statsmodels.api as sm
from io import StringIO

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Universal Statistical Analyzer",
    layout="wide"
)

# ===============================
# Custom Styling
# ===============================
st.markdown("""
<style>
.result-box {
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 10px;
}
.sig { background-color: #e8f5e9; border-left: 6px solid #2e7d32; }
.nsig { background-color: #ffebee; border-left: 6px solid #c62828; }
.info { background-color: #e3f2fd; border-left: 6px solid #1565c0; }
.warn { background-color: #fff8e1; border-left: 6px solid #f9a825; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Universal Statistical Analyzer (SPSS-Style)")
st.markdown(
    "<div class='result-box info'>"
    "✔ Supports SPSS (.sav), CSV, Excel, TSV, and Stata files<br>"
    "✔ Unicode-friendly statistical reporting<br>"
    "✔ Color-coded results interpretation"
    "</div>",
    unsafe_allow_html=True
)

# ===============================
# File Upload
# ===============================
uploaded_file = st.file_uploader(
    "📂 Upload Data File",
    type=["sav", "csv", "xlsx", "xls", "tsv", "dta"]
)

def load_data(file):
    ext = file.name.split(".")[-1].lower()

    if ext == "sav":
        df, meta = pyreadstat.read_sav(file)
        return df, meta
    elif ext == "csv":
        return pd.read_csv(file), None
    elif ext == "tsv":
        return pd.read_csv(file, sep="\t"), None
    elif ext in ["xlsx", "xls"]:
        return pd.read_excel(file), None
    elif ext == "dta":
        return pd.read_stata(file), None
    else:
        return None, None

if uploaded_file:
    df, meta = load_data(uploaded_file)

    st.success(f"✔ Loaded file: {uploaded_file.name}")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # ===============================
    # Data Preview
    # ===============================
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ===============================
    # Descriptive Statistics
    # ===============================
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe(include="all"))

    # ===============================
    # Normality Tests
    # ===============================
    st.subheader("📌 Normality Tests (Shapiro–Wilk)")

    normality_results = []

    for col in numeric_cols:
        if df[col].dropna().shape[0] >= 3:
            W, p = stats.shapiro(df[col].dropna())
            normality_results.append((col, W, p))

            box_class = "sig" if p > 0.05 else "warn"
            interpretation = "✓ Normal distribution" if p > 0.05 else "⚠ Deviates from normality"

            st.markdown(
                f"<div class='result-box {box_class}'>"
                f"<b>{col}</b> → W = {W:.3f}, p = {p:.4f}<br>"
                f"{interpretation} (α = 0.05)"
                "</div>",
                unsafe_allow_html=True
            )

    # ===============================
    # Independent Samples T-Test
    # ===============================
    st.subheader("🧪 Independent Samples t-Test")

    if len(numeric_cols) >= 2:
        v1 = st.selectbox("Group 1 Variable", numeric_cols)
        v2 = st.selectbox("Group 2 Variable", numeric_cols, index=1)

        t, p = stats.ttest_ind(df[v1].dropna(), df[v2].dropna(), equal_var=False)

        box = "sig" if p < 0.05 else "nsig"
        sig_text = "✓ Statistically significant difference" if p < 0.05 else "✗ No significant difference"

        st.markdown(
            f"<div class='result-box {box}'>"
            f"t = {t:.3f}, p = {p:.4f}<br>"
            f"{sig_text}"
            "</div>",
            unsafe_allow_html=True
        )

    # ===============================
    # Chi-Square Test
    # ===============================
    st.subheader("🧪 Chi-Square Test of Independence")

    if len(categorical_cols) >= 2:
        c1 = st.selectbox("Categorical Variable 1", categorical_cols)
        c2 = st.selectbox("Categorical Variable 2", categorical_cols, index=1)

        table = pd.crosstab(df[c1], df[c2])
        χ2, pχ, dof, _ = stats.chi2_contingency(table)

        box = "sig" if pχ < 0.05 else "nsig"
        result = "✓ Variables are associated" if pχ < 0.05 else "✗ No association found"

        st.markdown(
            f"<div class='result-box {box}'>"
            f"χ²({dof}) = {χ2:.3f}, p = {pχ:.4f}<br>"
            f"{result}"
            "</div>",
            unsafe_allow_html=True
        )

        st.dataframe(table)

    # ===============================
    # Linear Regression
    # ===============================
    st.subheader("📉 Simple Linear Regression")

    y = st.selectbox("Dependent Variable (Y)", numeric_cols, key="reg_y")
    x = st.selectbox("Independent Variable (X)", numeric_cols, index=1, key="reg_x")

    X = sm.add_constant(df[x])
    model = sm.OLS(df[y], X, missing="drop").fit()

    β = model.params[1]
    pβ = model.pvalues[1]

    box = "sig" if pβ < 0.05 else "nsig"

    st.markdown(
        f"<div class='result-box {box}'>"
        f"β = {β:.3f}, R² = {model.rsquared:.3f}, p = {pβ:.4f}<br>"
        f"{'✓ Significant predictor' if pβ < 0.05 else '✗ Not a significant predictor'}"
        "</div>",
        unsafe_allow_html=True
    )

    # ===============================
    # Unicode-Friendly Report
    # ===============================
    st.subheader("📝 Automatic Statistical Report")

    report = StringIO()
    report.write("STATISTICAL ANALYSIS REPORT\n")
    report.write("════════════════════════════\n\n")
    report.write(f"File: {uploaded_file.name}\n")
    report.write(f"Cases (N) = {df.shape[0]}, Variables = {df.shape[1]}\n\n")

    report.write("Descriptive Statistics:\n")
    for col in numeric_cols:
        report.write(
            f"{col}: μ = {df[col].mean():.2f}, σ = {df[col].std():.2f}\n"
        )

    report.write("\nRegression Analysis:\n")
    report.write(
        f"{x} → {y}: β = {β:.2f}, R² = {model.rsquared:.2f}, p = {pβ:.4f}\n"
    )

    report_text = report.getvalue()

    st.text_area("📄 Generated Report (Unicode-safe)", report_text, height=350)

    st.download_button(
        "📥 Download Report",
        report_text,
        "Statistical_Report_Unicode.txt",
        "text/plain"
    )
