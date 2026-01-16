import streamlit as st
import pandas as pd
import re
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Job Scope Analyser", layout="wide")

st.title("🎯 Job Scope Analyser")

# --- 1. SIDEBAR SETUP ---
st.sidebar.header("Configuration")

# File Uploader - This is the ONLY way data gets into the app
uploaded_file = st.sidebar.file_uploader("Upload your cleaned_data.csv", type=['csv'])

if uploaded_file:
    # Load data into RAM
    df_raw = pd.read_csv(uploaded_file)
    
    # Check for required columns
    required_cols = ['jobOpening_approvedAt', 'jobOpening_professionFinal', 'cleaned_text']
    if not all(col in df_raw.columns for col in required_cols):
        st.error(f"CSV must contain these columns: {required_cols}")
        st.stop()

    # Date Filters
    df_raw['jobOpening_approvedAt'] = pd.to_datetime(df_raw['jobOpening_approvedAt'], errors='coerce')
    start_date = st.sidebar.date_input("Start Date", df_raw['jobOpening_approvedAt'].min())
    end_date = st.sidebar.date_input("End Date", df_raw['jobOpening_approvedAt'].max())

    # Profession Selector
    all_profs = sorted(df_raw['jobOpening_professionFinal'].dropna().unique().tolist())
    target_profs = st.sidebar.multiselect("Select Professions", all_profs, default=all_profs[:2])

    # Blocklist
    blocklist_text = st.sidebar.text_area("Terms to ignore (Blocklist)", "to, and, the, a, bachelor, degree, years, experience")
    BLOCKLIST = set([x.strip().lower() for x in blocklist_text.split(',')])

    # --- 2. MODEL LOADING (Optimized) ---
    @st.cache_resource
    def load_model():
        # Using the specific model you requested
        return pipeline("ner", model="Nucha/Nucha_ITSkillNER_BERT", aggregation_strategy="simple")

    def clean_skill(skill_text):
        if not isinstance(skill_text, str): return None
        clean = skill_text.replace("##", "").strip().lower()
        clean = re.sub(r'[^a-z0-9\s\+\#\.]', '', clean)
        if clean in BLOCKLIST or len(clean) < 2 or clean.isdigit():
            return None
        return clean

    # --- 3. PROCESSING ---
    if st.button("🚀 Run Analysis"):
        # Filter data based on sidebar inputs
        mask = (df_raw['jobOpening_approvedAt'].dt.date >= start_date) & \
               (df_raw['jobOpening_approvedAt'].dt.date <= end_date) & \
               (df_raw['jobOpening_professionFinal'].isin(target_profs))
        df = df_raw[mask].copy()

        if df.empty:
            st.warning("No data found for the selected filters.")
        else:
            with st.spinner(f"Analyzing {len(df)} job descriptions..."):
                ner_model = load_model()
                
                def extract_skills(text):
                    try:
                        # Truncate to save memory
                        results = ner_model(str(text)[:1000])
                        return list(set([clean_skill(r['word']) for r in results if clean_skill(r['word'])]))
                    except: return []

                df['skills_found'] = df['cleaned_text'].apply(extract_skills)
                skills_exploded = df.explode('skills_found').dropna(subset=['skills_found'])

                # --- 4. DASHBOARD ---
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Top Skills Found")
                    top_skills = skills_exploded['skills_found'].value_counts().head(20)
                    st.bar_chart(top_skills)
                
                with col2:
                    st.subheader("Data Summary")
                    st.write(f"Total Jobs Analyzed: {len(df)}")
                    st.write(f"Unique Skills Identified: {skills_exploded['skills_found'].nunique()}")

                # Download Results
                csv_data = skills_exploded.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Analysis (CSV)", csv_data, "job_analysis.csv", "text/csv")
else:
    st.info("👋 Welcome! Please upload your CSV file in the sidebar to start.")