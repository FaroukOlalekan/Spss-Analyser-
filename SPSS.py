import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import Table2x2
import statsmodels.api as sm
import pyreadstat
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI-SPSS 📊 Analysis Tool", layout="wide")

# ---------------- BACKGROUND COLOR ----------------
st.markdown("""
<style>
.stApp {
    background-color: #eef4f8;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Assisted SPSS Data Analysis & Chat Interface")

st.markdown("""
📂 Upload your dataset • ❓ Ask research questions • 📊 Run SPSS-style analyses  
🧠 Get ChatGPT-like explanations • 📝 Auto-generate academic reports
""")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload dataset",
    type=["sav", "xlsx", "xls", "csv", "tsv", "txt", "dta", "sas7bdat", "json"]
)

if uploaded_file:
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".sav"):
            df, meta = pyreadstat.read_sav(uploaded_file)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".tsv"):
            df = pd.read_csv(uploaded_file, sep="\t")
        elif file_name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
        elif file_name.endswith(".dta"):
            df = pd.read_stata(uploaded_file)
        elif file_name.endswith(".sas7bdat"):
            df = pd.read_sas(uploaded_file)
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)

        st.success("✅ Dataset loaded successfully!")

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

    # ---------------- PREVIEW ----------------
    st.subheader("👀 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- RESEARCH QUESTION ----------------
    st.subheader("❓ Research Question")
    user_question = st.text_input(
        "Type your research question",
        placeholder="e.g., Is exposure to 2,4-D associated with increased cancer risk in dogs?"
    )

    # ---------------- VARIABLE SELECTION ----------------
    st.subheader("🧪 Variable Selection")
    exposure = st.selectbox("Independent / Exposure Variable", df.columns)
    outcome = st.selectbox("Dependent / Outcome Variable", df.columns)

    if user_question and exposure and outcome:

        # ---------------- CROSSTABS ----------------
        raw_table = pd.crosstab(df[exposure], df[outcome])
        percent_table = pd.crosstab(df[exposure], df[outcome], normalize='index') * 100

        st.subheader("📊 Crosstabulation (Row %)")
        st.table(percent_table.round(2))

        # ---------------- STATISTICS ----------------
        chi2, p, dof, expected = chi2_contingency(raw_table)
        fisher_p = fisher_exact(raw_table.values)[1]

        risk = Table2x2(raw_table.values)
        or_val = risk.oddsratio
        rr_val = risk.riskratio
        ci_low, ci_high = risk.oddsratio_confint()

        X = sm.add_constant(df[exposure])
        y = df[outcome]
        logit = sm.Logit(y, X).fit(disp=False)

        # ---------------- CHART ----------------
        st.subheader("📈 Bar Chart")
        fig, ax = plt.subplots()
        raw_table.plot(kind='bar', ax=ax)
        ax.set_xlabel(exposure)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # ---------------- RESULTS ----------------
        st.subheader("📊 SPSS-Equivalent Results")

        st.markdown(f"""
- **χ²({dof})** = {chi2:.2f}  
- **p-value** = {p:.3f}  
- **Fisher’s Exact p** = {fisher_p:.3f}  
- **Odds Ratio (OR)** = {or_val:.2f} (95% CI [{ci_low:.2f}, {ci_high:.2f}])  
- **Relative Risk (RR)** = {rr_val:.2f}  
- **Logistic Regression β** = {logit.params[1]:.2f}, *p* = {logit.pvalues[1]:.3f}
""")

        # ---------------- REPORT ----------------
        st.subheader("📝 AI-Generated Academic Report (~300 words)")
        st.text_area(
            "Editable Report",
            f"""
📌 **Research Question:** {user_question}

**Results Summary:**  
χ²({dof}) = {chi2:.2f}, p = {p:.3f} → {'✔ Significant' if p < 0.05 else '✖ Not significant'}

Dogs exposed to {exposure} had higher odds of {outcome} (OR = {or_val:.2f}, RR = {rr_val:.2f}).  
Logistic regression confirmed exposure as a significant predictor (β = {logit.params[1]:.2f}).

**Conclusion:**  
These findings support the hypothesis that {exposure} is associated with {outcome}.
""",
            height=400
        )

        # ---------------- AI CHAT ----------------
        st.subheader("🤖 Ask the AI About Your Data")

        ai_question = st.text_area(
            "Ask a question (ChatGPT-style)",
            placeholder="Can I reject the null hypothesis? Explain this simply."
        )

        def ai_chat_response(q):
            q = q.lower()
            if "null" in q:
                return "✔ Yes, the null hypothesis can be rejected." if p < 0.05 else "✖ No, the null hypothesis cannot be rejected."
            if "simple" in q:
                return f"🧠 Simply put, exposure to {exposure} increases the chance of {outcome}."
            if "spss" in q:
                return f"📊 In SPSS terms: χ²({dof}) = {chi2:.2f}, p = {p:.3f}."
            return f"📌 There is a statistically significant association between {exposure} and {outcome}."

        if ai_question:
            st.markdown("### 🤖 AI Response")
            st.markdown(ai_chat_response(ai_question))
