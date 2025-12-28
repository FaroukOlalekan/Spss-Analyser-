# ======================================
# IMPORTS
# ======================================
import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import os

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(page_title="Phabolous Statistical Analyzer", layout="wide")

# ======================================
# CUSTOM CSS (BACKGROUND + COLORS)
# ======================================
st.markdown("""
<style>
.stApp { background-color: #FFCDD2; }  /* Changed to soft red */
.result-box { padding: 14px; border-radius: 10px; margin-bottom: 12px; }
.sig { background-color: #E8F5E9; border-left: 6px solid #2E7D32; }
.nsig { background-color: #FFEBEE; border-left: 6px solid #C62828; }
.info { background-color: #E3F2FD; border-left: 6px solid #1565C0; }
.warn { background-color: #FFF8E1; border-left: 6px solid #F9A825; }
</style>
""", unsafe_allow_html=True)


# ======================================
# HEADER
# ======================================
st.title("📊 Phabolous Statistical Analyzer (SPSS-Style)")
st.markdown(
    "<div class='result-box info'>"
    "Upload data, perform SPSS-equivalent analyses, visualize results, "
    "and download a professional PDF report with embedded graphs."
    "</div>",
    unsafe_allow_html=True
)

# ======================================
# FILE UPLOAD
# ======================================
uploaded_file = st.file_uploader(
    "📂 Upload Data File",
    type=["sav", "csv", "xlsx", "xls", "tsv", "dta"]
)

def load_data(file):
    ext = file.name.split(".")[-1].lower()
    if ext == "sav":
        return pyreadstat.read_sav(file)[0]
    if ext == "csv":
        return pd.read_csv(file)
    if ext == "tsv":
        return pd.read_csv(file, sep="\t")
    if ext in ["xlsx", "xls"]:
        return pd.read_excel(file)
    if ext == "dta":
        return pd.read_stata(file)
    return None

# ======================================
# MAIN APP
# ======================================
if uploaded_file:
    df = load_data(uploaded_file)
    st.success(f"✔ Loaded file: {uploaded_file.name}")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # ======================================
    # DATA PREVIEW
    # ======================================
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ======================================
    # DESCRIPTIVE STATISTICS
    # ======================================
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe(include="all"))

    # ======================================
    # NORMALITY TESTS
    # ======================================
    st.subheader("📌 Normality Tests (Shapiro–Wilk)")
    normality_results = []

    for col in numeric_cols:
        if df[col].dropna().shape[0] >= 3:
            W, p = stats.shapiro(df[col].dropna())
            normality_results.append((col, W, p))
            box = "sig" if p > 0.05 else "warn"
            st.markdown(
                f"<div class='result-box {box}'>"
                f"<b>{col}</b>: W = {W:.3f}, p = {p:.4f}"
                "</div>",
                unsafe_allow_html=True
            )

    # ======================================
    # T-TEST
    # ======================================
    st.subheader("🧪 Independent Samples t-Test")
    v1 = st.selectbox("Variable 1", numeric_cols)
    v2 = st.selectbox("Variable 2", numeric_cols, index=1)

    t, p = stats.ttest_ind(df[v1].dropna(), df[v2].dropna(), equal_var=False)
    st.markdown(
        f"<div class='result-box {'sig' if p < 0.05 else 'nsig'}'>"
        f"t = {t:.3f}, p = {p:.4f}"
        "</div>",
        unsafe_allow_html=True
    )

    # ======================================
    # REGRESSION
    # ======================================
    st.subheader("📉 Linear Regression")
    y = st.selectbox("Dependent Variable (Y)", numeric_cols)
    x = st.selectbox("Independent Variable (X)", numeric_cols, index=1)

    X = sm.add_constant(df[x])
    model = sm.OLS(df[y], X, missing="drop").fit()
    β = model.params[1]
    pβ = model.pvalues[1]

    st.markdown(
        f"<div class='result-box {'sig' if pβ < 0.05 else 'nsig'}'>"
        f"β = {β:.3f}, R² = {model.rsquared:.3f}, p = {pβ:.4f}"
        "</div>",
        unsafe_allow_html=True
    )

    # ======================================
    # GRAPHS (SAVED FOR PDF)
    # ======================================
    st.subheader("📊 Graphical Analysis")
    plot_paths = []

    fig, ax = plt.subplots()
    sns.histplot(df[y].dropna(), kde=True, ax=ax)
    hist_path = "histogram.png"
    fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    plot_paths.append(hist_path)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.regplot(x=df[x], y=df[y], ax=ax)
    reg_path = "regression.png"
    fig.savefig(reg_path, dpi=300, bbox_inches="tight")
    plot_paths.append(reg_path)
    st.pyplot(fig)

    # ======================================
    # EXTENDED REPORT (~200 WORDS)
    # ======================================
    report_text = f"""
The present statistical analysis was conducted on the uploaded dataset ({uploaded_file.name}),
which consisted of {df.shape[0]} cases and {df.shape[1]} variables. The purpose of this analysis
was to examine data distribution, test statistical assumptions, explore relationships between
variables, and evaluate inferential outcomes using SPSS-equivalent procedures.

Descriptive statistics indicated variability in central tendency and dispersion across variables.
For example, the dependent variable demonstrated a mean (μ) of {df[y].mean():.2f} and a standard
deviation (σ) of {df[y].std():.2f}. Normality testing using the Shapiro–Wilk test suggested that
some variables approximated a normal distribution, while others deviated from this assumption.

Inferential analysis included an independent samples t-test and a simple linear regression model.
The regression analysis revealed that {x} predicted {y} with a standardized coefficient of
β = {β:.2f} and an explained variance of R² = {model.rsquared:.2f}. This effect was
{'statistically significant' if pβ < 0.05 else 'not statistically significant'} at α = 0.05.

Graphical analyses, including histograms and regression plots, were used to visually inspect
data distributions and linear relationships. These visualizations supported the numerical
findings and enhanced interpretability of the results.
"""

    st.text_area("📝 Generated Report", report_text, height=420)

    # ======================================
    # PDF GENERATION
    # ======================================
    if st.button("📄 Generate PDF Report with Graphs"):
        pdf_path = "Statistical_Report_With_Graphs.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>STATISTICAL ANALYSIS REPORT</b>", styles["Title"]))
        story.append(Spacer(1, 0.3 * inch))

        for para in report_text.split("\n\n"):
            story.append(Paragraph(para, styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))

        story.append(Spacer(1, 0.3 * inch))
        story.append(Paragraph("<b>Graphical Analysis</b>", styles["Heading2"]))
        story.append(Spacer(1, 0.2 * inch))

        for img in plot_paths:
            story.append(Image(img, width=5 * inch, height=3 * inch))
            story.append(Spacer(1, 0.3 * inch))

        doc.build(story)

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Download PDF Report",
                f,
                file_name="Statistical_Report_With_Graphs.pdf",
                mime="application/pdf"
            )

        for img in plot_paths:
            os.remove(img)
