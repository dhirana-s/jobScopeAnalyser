import streamlit as st
import pandas as pd
import re
import html
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Job Scope Analyser", layout="wide")

# --- PASSWORD CHECK ---
def check_password():
    """Returns True if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    placeholder = st.empty()
    with placeholder.container():
        st.write("## 🔒 Access Restricted")
        password = st.text_input("Enter the password to use this tool", type="password")
        if st.button("Log In"):
            if password == st.secrets["password"]:
                st.session_state["password_correct"] = True
                placeholder.empty()
                st.rerun()
            else:
                st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()

# --- MAIN APP UI ---
st.title("🎯 Job Scope Analyser")
st.markdown("""
**Privacy Note:** This tool is an 'Empty Shell'. No data is stored on this server. 
Analysis happens in temporary memory and is wiped when you close the browser tab.
""")

# --- 1. CONFIGURATION (SIDEBAR) ---
st.sidebar.header("Configuration")

# File Uploader (Supports CSV and Excel)
uploaded_file = st.sidebar.file_uploader("Upload data", type=['csv', 'xlsx'])

if uploaded_file:
    # Load Data
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)
    
    # Check Columns
    # Note: I used the column names from your notes to be safe!
    required_cols = ['jobOpening_approvedAt', 'jobOpening_professionFinal', 'jobOpening_jobScope']
    if not all(col in df_raw.columns for col in required_cols):
        st.error(f"File must contain these columns: {required_cols}")
        st.stop()

    # Date Filter
    df_raw['jobOpening_approvedAt'] = pd.to_datetime(df_raw['jobOpening_approvedAt'], errors='coerce')
    min_date = df_raw['jobOpening_approvedAt'].min()
    max_date = df_raw['jobOpening_approvedAt'].max()
    
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)

    # Profession Filter
    all_profs = sorted(df_raw['jobOpening_professionFinal'].dropna().unique().tolist())
    target_profs = st.sidebar.multiselect("Select Professions", all_profs, default=all_profs[:2])

    # Blocklist
    blocklist_text = st.sidebar.text_area("Blocklist (Terms to Ignore)", "to, and, the, a, bachelor, degree, years, experience, strong, ability, working, knowledge")
    BLOCKLIST = set([x.strip().lower() for x in blocklist_text.split(',')])

    # --- 2. ADVANCED CLEANING (From your notes!) ---
    def clean_text_advanced(raw_text):
        if not isinstance(raw_text, str): return ""
        # 1. HTML Unescape (from your notes)
        text = html.unescape(raw_text)
        # 2. Remove HTML tags (from your notes)
        text = re.sub(r'<.*?>', ' ', text)
        # 3. Standardize whitespace
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    # --- 3. MODEL SETUP ---
    @st.cache_resource
    def load_model():
        return pipeline("ner", model="Nucha/Nucha_ITSkillNER_BERT", aggregation_strategy="simple")

    def post_process_skill(skill_text):
        """Final polish on the extracted skill word"""
        clean = skill_text.replace("##", "").strip().lower()
        clean = re.sub(r'[^a-z0-9\s\+\#\.]', '', clean) # Preserve C++, C#, Node.js
        if clean in BLOCKLIST or len(clean) < 2 or clean.isdigit():
            return None
        return clean

    # --- 4. EXECUTION ---
    if st.button("🚀 Run Analysis"):
        # Apply Filters
        mask = (df_raw['jobOpening_approvedAt'].dt.date >= start_date) & \
               (df_raw['jobOpening_approvedAt'].dt.date <= end_date) & \
               (df_raw['jobOpening_professionFinal'].isin(target_profs))
        df = df_raw[mask].copy()

        if df.empty:
            st.warning("No data found for these filters.")
        else:
            with st.spinner("Cleaning text and Extracting skills..."):
                # A. Apply your Advanced Cleaning
                df['cleaned_text'] = df['jobOpening_jobScope'].apply(clean_text_advanced)
                
                # B. Run BERT
                ner_model = load_model()
                
                def extract_skills(text):
                    try:
                        results = ner_model(text[:1000])
                        skills = [post_process_skill(r['word']) for r in results]
                        return list(set([s for s in skills if s]))
                    except: return []

                df['skills_found'] = df['cleaned_text'].apply(extract_skills)
                
                # C. Explode for Charts
                skills_exploded = df.explode('skills_found').dropna(subset=['skills_found'])

                # --- 5. VISUALIZATION ---
                st.success(f"Analysis Complete! Found {skills_exploded['skills_found'].nunique()} unique skills.")

                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Skills (Interactive)")
                    # Simple bar chart is built-in to Streamlit
                    top_skills = skills_exploded['skills_found'].value_counts().head(20)
                    st.bar_chart(top_skills)

                with col2:
                    st.subheader("Skills by Seniority")
                    # Using the column name from your notes: 'jobOpening_workExperienceYearsFinal'
                    if 'jobOpening_workExperienceYearsFinal' in df.columns:
                        seniority_dist = skills_exploded.groupby('jobOpening_workExperienceYearsFinal')['skills_found'].count()
                        st.bar_chart(seniority_dist)
                    else:
                        st.info("No seniority column found.")

                # Download Button
                csv_data = skills_exploded.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Full CSV Report", csv_data, "skill_analysis.csv", "text/csv")

else:
    st.info("👋 Upload a CSV or Excel file to begin.")