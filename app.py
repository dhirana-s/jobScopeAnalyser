# import streamlit as st
# import polars as pl
# import pandas as pd
# import plotly.express as px
# import os

# # --- 1. APP CONFIGURATION ---
# st.set_page_config(page_title="Career Skill Evolution", layout="wide", page_icon="🚀")

# # Styling to make it look modern
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f5f7f9;
#     }
#     .stMetric {
#         background-color: #ffffff;
#         padding: 15px;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.05);
#     }
#     </style>
#     """, unsafe_allow_html=True)

# st.title("🚀 Tech Career Skill Evolution Dashboard")
# st.markdown("""
# This dashboard visualizes how skill requirements shift from **Junior** to **Mid-level** and finally **Senior** roles 
# based on job description analysis.
# """)

# # --- 2. DATA LOADING ---
# @st.cache_data
# def load_data():
#     # Paths to the reports generated in 3_analysis.py
#     swe_path = "reports/swe_skill_evolution_full.csv"
#     ai_path = "reports/ai_skill_evolution_full.csv"
    
#     if not os.path.exists(swe_path) or not os.path.exists(ai_path):
#         st.error("Report files not found! Please run 3_analysis.py first.")
#         st.stop()
        
#     swe_df = pl.read_csv(swe_path)
#     ai_df = pl.read_csv(ai_path)
#     return {"SOFTWARE_ENGINEER": swe_df, "AI_ENGINEER": ai_df}

# data_dict = load_data()

# # --- 3. SIDEBAR CONTROLS ---
# st.sidebar.header("Filter Settings")
# selected_prof = st.sidebar.selectbox("Select Profession", list(data_dict.keys()))
# top_n = st.sidebar.slider("Number of Skills to Show", 10, 100, 50)

# # Filter data based on sidebar selection
# full_df = data_dict[selected_prof]
# df_display = full_df.head(top_n)

# # --- 4. DATA VISUALIZATION: THE TRAJECTORY ---
# st.subheader(f"📈 Skill Importance Trajectory: {selected_prof}")

# # Select a skill to highlight from the top list
# target_skill = st.selectbox("Pick a skill to analyze its career path:", full_df["skills_found"].to_list())

# # Prepare data for line chart
# skill_row = full_df.filter(pl.col("skills_found") == target_skill)

# if not skill_row.is_empty():
#     plot_df = pd.DataFrame({
#         "Career Level": ["Junior", "Mid-level", "Senior"],
#         "Market Share (%)": [
#             skill_row["Junior"][0], 
#             skill_row["Mid-level"][0], 
#             skill_row["Senior"][0]
#         ]
#     })

#     fig = px.line(
#         plot_df, 
#         x="Career Level", 
#         y="Market Share (%)", 
#         markers=True,
#         text=[f"{v:.1f}%" for v in plot_df["Market Share (%)"]],
#         title=f"Trend for '{target_skill}'"
#     )
    
#     fig.update_traces(textposition="top center", line_color="#007bff", marker_size=10)
#     fig.update_layout(yaxis_range=[0, max(plot_df["Market Share (%)"]) * 1.2])
    
#     st.plotly_chart(fig, use_container_width=True)

# # --- 5. THE COMPARISON TABLE ---
# st.subheader("🔍 Raw Metrics (Top 100 Skills)")
# st.dataframe(
#     df_display.to_pandas().style.background_gradient(subset=["Total_Rel_Change_Pct"], cmap="RdYlGn"),
#     use_container_width=True,
#     height=400
# )

# # --- 6. STRATEGIC INSIGHTS ---
# st.divider()
# st.subheader("💡 Strategic Career Insights")
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.info("📉 **Gatekeepers**")
#     # Skills high in Junior but falling sharply in Senior
#     gatekeepers = full_df.filter(pl.col("Junior_to_Senior_Diff") < -1.5).sort("Junior", descending=True).head(5)
#     st.write("Mandatory for entry, but mention frequency drops as you promote:")
#     for s in gatekeepers["skills_found"].to_list():
#         st.markdown(f"- {s}")

# with col2:
#     st.success("📈 **Differentiators**")
#     # Skills rising towards Senior
#     differentiators = full_df.filter(pl.col("Junior_to_Senior_Diff") > 0.5).sort("Senior", descending=True).head(5)
#     st.write("These skills become significantly more critical for leadership roles:")
#     for s in differentiators["skills_found"].to_list():
#         st.markdown(f"- {s}")

# with col3:
#     st.warning("🌉 **Bridge Skills**")
#     # Skills that peak at Mid-level
#     bridge_skills = full_df.filter(
#         (pl.col("Mid-level") > pl.col("Junior")) & 
#         (pl.col("Mid-level") > pl.col("Senior"))
#     ).sort("Mid-level", descending=True).head(5)
#     st.write("These represent the peak of hands-on technical execution:")
#     for s in bridge_skills["skills_found"].to_list():
#         st.markdown(f"- {s}")






























# # FINAL WORKING VERSION
# import streamlit as st
# import polars as pl
# import pandas as pd
# import plotly.express as px
# import os
# import re
# import time
# from thefuzz import process, fuzz
# from bs4 import BeautifulSoup
# from transformers import pipeline

# # --- 1. APP CONFIGURATION & STYLING ---
# st.set_page_config(page_title="Career Skill Evolution Pro", layout="wide", page_icon="🚀")

# st.markdown("""
#     <style>
#     .main { background-color: #f5f7f9; }
#     .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
#     footer {visibility: hidden;}
#     </style>
#     """, unsafe_allow_html=True)

# # --- 2. PIPELINE ENGINES (Your Core Logic) ---

# @st.cache_resource
# def load_bert_model():
#     """Loads the BERT NER model once and keeps it in memory."""
#     return pipeline(
#         "ner", 
#         model="Nucha/Nucha_ITSkillNER_BERT", 
#         aggregation_strategy="simple",
#         device=-1 
#     )

# def clean_html_to_text(html):
#     if not html: return ""
#     return BeautifulSoup(html, "html.parser").get_text(separator=" ")

# def clean_extracted_token(skill_text):
#     if not skill_text: return None
#     clean = skill_text.replace("##", "").strip().lower()
#     clean = re.sub(r'[^a-z0-9\s\+\#\.]', '', clean)
#     if len(clean) < 2 or clean.isdigit(): return None
#     return " ".join(clean.split())

# def run_full_pipeline(uploaded_file, skill_extractor):
#     """Executes Stage 1, 2, and 3 sequentially."""
    
#     # --- STAGE 1: CLEANING ---
#     df = pl.read_excel(uploaded_file, sheet_name="jobApplications_FY24", engine="calamine")
#     df = df.with_columns(
#         pl.col("jobOpening_profession").str.replace(r"SOFTWARE_ENGINEER_.*", "SOFTWARE_ENGINEER"),
#         pl.col("jobOpening_workExperienceYears").replace({
#             "NO_EXPERIENCE": "Junior", "ONE_TWO_YEARS": "Junior",
#             "THREE_FIVE_YEARS": "Mid-level", "SIX_TEN_YEARS": "Senior", "GREATER_TEN_YEARS": "Senior"
#         })
#     )
    
#     # --- STAGE 2: PREPROCESSING ---
#     PROFESSION_COL = 'jobOpening_professionFinal'
#     TARGET_PROFESSIONS = ['SOFTWARE_ENGINEER', 'AI_ENGINEER']
    
#     if "jobOpening_profession" in df.columns:
#         df = df.rename({"jobOpening_profession": PROFESSION_COL})
    
#     df = df.filter(pl.col(PROFESSION_COL).is_in(TARGET_PROFESSIONS)).unique(subset=["jobOpening_serialNumber"])
#     df = df.with_columns(pl.col("jobOpening_jobScope").map_elements(clean_html_to_text, return_dtype=pl.String).alias("clean_text"))
    
#     # BERT Extraction
#     unique_texts_df = df.select("clean_text").unique().drop_nulls()
#     unique_text_list = unique_texts_df["clean_text"].to_list()
#     unique_results = []

#     for text in unique_text_list:
#         words = text.split()
#         chunks = [" ".join(words[i:i + 300]) for i in range(0, len(words), 300)]
#         aggregated_skills = set()
#         for chunk in chunks:
#             results = skill_extractor(chunk)
#             for res in results:
#                 if res['entity_group'] in ['HSKILL', 'SSKILL']:
#                     skill = clean_extracted_token(res['word'])
#                     if skill: aggregated_skills.add(skill)
#         unique_results.append(list(aggregated_skills))

#     unique_mapping_df = unique_texts_df.with_columns(pl.Series("raw_skills", unique_results))
#     df = df.join(unique_mapping_df, on="clean_text", how="left")

#     # Fuzzy Deduplication
#     all_found = df.select(pl.col("raw_skills").flatten()).unique().to_series().to_list()
#     all_found = sorted([s for s in all_found if s], key=len)
#     mapping, processed = {}, set()
#     for skill in all_found:
#         if skill in processed: continue
#         processed.add(skill)
#         mapping[skill] = skill
#         candidates = [c for c in all_found if c not in processed]
#         if not candidates: continue
#         match = process.extractOne(skill, candidates, scorer=fuzz.ratio)
#         if match and match[1] >= 92:
#             mapping[match[0]] = skill
#             processed.add(match[0])

#     patterns = (
#         df.explode("raw_skills")
#         .with_columns(pl.col("raw_skills").replace(mapping).alias("skills_found"))
#         .drop_nulls("skills_found")
#         .group_by([PROFESSION_COL, "jobOpening_workExperienceYears", "skills_found"])
#         .len()
#     )

#     # --- STAGE 3: ANALYSIS ---
#     STOP_WORDS = ["and the", "with", "are", "er", "less", "inform", "aw", "and", "the", "of", "to", "in", "for", "on", "at", "by", "from", "as", "with a", "and an", "full", "natural", "technical", "motivated", "command", "computing and", "solving skills and", "skills", "attention", "organization", "is", "be", "an", "plus", "red", "deep", "native", "experience"]
    
#     patterns = patterns.with_columns(
#         pl.col("skills_found").replace({
#             "bachelor": "bachelor's degree", "s degree": "bachelor's degree", "bachelor s": "bachelor's degree",
#             "master": "master's degree", "master s": "master's degree", "master s degree": "master's degree",
#             "apis": "api", "c + +": "c++", "c +": "c++", "react. js": "react", "reactjs": "react",
#             "robotic": "robotics", "machine vision": "computer vision", "verbal communication": "communication",
#             "effective communication": "communication", "communication skills": "communication",
#             "ml": "machine learning", "gging": "debugging", "native": "react native"
#         })
#     ).filter(~pl.col("skills_found").is_in(STOP_WORDS))

#     patterns = patterns.group_by([PROFESSION_COL, "jobOpening_workExperienceYears", "skills_found"]).agg(pl.col("len").sum())
    
#     category_totals = patterns.group_by([PROFESSION_COL, "jobOpening_workExperienceYears"]).agg(pl.sum("len").alias("total_mentions_at_level"))
#     df_norm = patterns.join(category_totals, on=[PROFESSION_COL, "jobOpening_workExperienceYears"])
#     df_norm = df_norm.with_columns((pl.col("len") / pl.col("total_mentions_at_level") * 100).alias("share_pct"))

#     def get_career_evolution(prof):
#         levels = ["Junior", "Mid-level", "Senior"]
#         js_df = df_norm.filter((pl.col(PROFESSION_COL) == prof) & (pl.col("jobOpening_workExperienceYears").is_in(levels)))
#         pivot = js_df.pivot(values="share_pct", index="skills_found", on="jobOpening_workExperienceYears", aggregate_function="sum").fill_null(0)
#         for col in levels:
#             if col not in pivot.columns: pivot = pivot.with_columns(pl.lit(0.0).alias(col))
        
#         pivot = pivot.with_columns(
#             (pl.col("Mid-level") - pl.col("Junior")).alias("Junior_to_Mid_Diff"),
#             (pl.col("Senior") - pl.col("Mid-level")).alias("Mid_to_Senior_Diff"),
#             (pl.col("Senior") - pl.col("Junior")).alias("Junior_to_Senior_Diff"),
#             pl.when(pl.col("Junior") == 0).then(pl.lit(100.0)).otherwise(((pl.col("Senior") - pl.col("Junior")) / pl.col("Junior")) * 100).alias("Total_Rel_Change_Pct")
#         )
#         return pivot.select(["skills_found", "Junior", "Mid-level", "Senior", "Junior_to_Mid_Diff", "Mid_to_Senior_Diff", "Junior_to_Senior_Diff", "Total_Rel_Change_Pct"]).sort("Junior", descending=True).head(100)

#     return {"SOFTWARE_ENGINEER": get_career_evolution("SOFTWARE_ENGINEER"), "AI_ENGINEER": get_career_evolution("AI_ENGINEER")}

# # --- 3. UI LAYOUT ---

# st.title("🚀 Tech Career Skill Evolution Tool")
# st.markdown("Upload your DTC Data file to analyze market trajectories from Junior to Senior levels.")

# with st.sidebar:
#     st.header("1. Data Upload")
#     uploaded_file = st.file_uploader("Choose Excel File", type="xlsx")
    
#     if st.button("🔄 Reset System"):
#         st.session_state.clear()
#         st.rerun()

# # --- 4. EXECUTION CONTROLS ---
# if uploaded_file and 'final_data' not in st.session_state:
#     if st.button("⚡ Run Full Analysis Pipeline"):
#         # Load model first (cached)
#         with st.spinner("Initializing AI Model (First run may take a minute)..."):
#             skill_extractor = load_bert_model()
            
#         with st.status("🛠️ Processing Career Pipeline...", expanded=True) as status:
#             st.write("Step 1: Cleaning & Experience Mapping...")
#             time.sleep(1) # Visual feedback
            
#             st.write("Step 2: BERT Skill Extraction (NER) & Fuzzy Matching...")
#             # This is the heavy part
#             results = run_full_pipeline(uploaded_file, skill_extractor)
            
#             st.write("Step 3: Normalizing & Strategic Analysis...")
#             st.session_state['final_data'] = results
#             status.update(label="✅ Analysis Complete!", state="complete")
#         st.balloons()

# # --- 5. DASHBOARD VISUALIZATION ---
# if 'final_data' in st.session_state:
#     data_dict = st.session_state['final_data']
    
#     st.sidebar.divider()
#     st.sidebar.header("2. Dashboard Filters")
#     selected_prof = st.sidebar.selectbox("Select Profession", list(data_dict.keys()))
#     top_n = st.sidebar.slider("Number of Skills to Show", 10, 100, 50)
    
#     full_df = data_dict[selected_prof]
#     df_display = full_df.head(top_n)

#     # Line Chart
#     st.subheader(f"📈 Skill Trajectory: {selected_prof}")
#     target_skill = st.selectbox("Analyze Career Path for Skill:", full_df["skills_found"].to_list())
    
#     skill_row = full_df.filter(pl.col("skills_found") == target_skill)
#     if not skill_row.is_empty():
#         plot_df = pd.DataFrame({
#             "Level": ["Junior", "Mid-level", "Senior"],
#             "Market Share (%)": [skill_row["Junior"][0], skill_row["Mid-level"][0], skill_row["Senior"][0]]
#         })
#         fig = px.line(plot_df, x="Level", y="Market Share (%)", markers=True, 
#                       text=[f"{v:.1f}%" for v in plot_df["Market Share (%)"]], title=f"Trend for '{target_skill}'")
#         fig.update_traces(textposition="top center", line_color="#007bff", marker_size=10)
#         fig.update_layout(yaxis_range=[0, max(plot_df["Market Share (%)"]) * 1.3])
#         st.plotly_chart(fig, use_container_width=True)

#     # Strategic Insights
#     st.divider()
#     st.subheader("💡 Strategic Insights")
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.info("📉 **Gatekeepers**")
#         gatekeepers = full_df.filter(pl.col("Junior_to_Senior_Diff") < -1.5).sort("Junior", descending=True).head(5)
#         for s in gatekeepers["skills_found"].to_list(): st.markdown(f"- {s}")

#     with col2:
#         st.success("📈 **Differentiators**")
#         diffs = full_df.filter(pl.col("Junior_to_Senior_Diff") > 0.5).sort("Senior", descending=True).head(5)
#         for s in diffs["skills_found"].to_list(): st.markdown(f"- {s}")

#     with col3:
#         st.warning("🌉 **Bridge Skills**")
#         bridges = full_df.filter((pl.col("Mid-level") > pl.col("Junior")) & (pl.col("Mid-level") > pl.col("Senior"))).sort("Mid-level", descending=True).head(5)
#         for s in bridges["skills_found"].to_list(): st.markdown(f"- {s}")

#     # Data Table
#     st.subheader("🔍 Full Evolution Matrix")
#     st.dataframe(df_display.to_pandas().style.background_gradient(subset=["Total_Rel_Change_Pct"], cmap="RdYlGn"), use_container_width=True)

# else:
#     st.info("👋 Ready to start. Please upload your dataset in the sidebar and run the analysis.")























































































# SNOWFLAKE
import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import time
from transformers import pipeline

# Import your custom modules
import cleaning
import preprocessing
import analysis

# --- 1. APP CONFIGURATION & STYLING ---
st.set_page_config(page_title="Career Skill Evolution Pro", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_bert_model():
    """Loads the BERT NER model once and keeps it in memory."""
    return pipeline(
        "ner", 
        model="Nucha/Nucha_ITSkillNER_BERT", 
        aggregation_strategy="simple",
        device=-1 
    )

# # --- 3. PIPELINE ORCHESTRATOR ---
# def run_full_pipeline(uploaded_file, skill_extractor, date_range):
#     """Executes the modular pipeline sequentially."""
    
#     # Stage 1: Cleaning & Date Filtering
#     cleaned_df = cleaning.run_cleaning(uploaded_file, date_range)
    
#     # Stage 2: Preprocessing (BERT extraction & Fuzzy matching)
#     patterns = preprocessing.run_preprocessing(cleaned_df, skill_extractor)
    
#     # Stage 3: Analysis (Career Evolution Metrics)
#     results = analysis.run_analysis(patterns)
    
#     return results


def run_full_pipeline(uploaded_file, skill_extractor, date_range):
    # Stage 1: Returns TWO items now
    cleaned_df, true_totals = cleaning.run_cleaning(uploaded_file, date_range)
    
    # Stage 2: Preprocessing
    patterns = preprocessing.run_preprocessing(cleaned_df, skill_extractor)
    
    # Stage 3: Analysis now uses the true_totals
    results = analysis.run_analysis(patterns, true_totals)
    
    return results

# --- 4. UI LAYOUT ---
st.title("🚀 Job Posting Analysis Tool")
st.markdown("Analyze role scope trajectories from Junior to Senior levels based on job posting data.")

with st.sidebar:
    st.header("1. Data Upload & Filters")
    uploaded_file = st.file_uploader("Choose Excel File (dtcExtract.xlsx)", type="xlsx")
    
    # Date Range Filter
    selected_dates = st.date_input(
        "Application Date Range",
        value=(pd.to_datetime("2024-01-01"), pd.to_datetime("2024-12-31")),
        format="DD/MM/YYYY"
    )
    
    st.divider()
    if st.button("🔄 Reset System", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# --- 5. EXECUTION CONTROLS ---
if uploaded_file and 'final_data' not in st.session_state:
    if st.button("⚡ Run Full Analysis Pipeline", use_container_width=True):
        
        # Load model first
        with st.spinner("Initializing AI Model (First run may take a minute)..."):
            skill_extractor = load_bert_model()
            
        # Run Pipeline with Status Updates
        with st.status("🛠️ Processing Career Pipeline...", expanded=True) as status:
            st.write("Step 1: Cleaning & Experience Mapping...")
            # Pipeline starts here
            results = run_full_pipeline(uploaded_file, skill_extractor, selected_dates)
            
            st.session_state['final_data'] = results
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
            
        st.balloons()

# --- 6. DASHBOARD VISUALIZATION ---
if 'final_data' in st.session_state:
    data_dict = st.session_state['final_data']
    
    st.sidebar.header("2. Dashboard Filters")
    selected_prof = st.sidebar.selectbox("Select Profession", list(data_dict.keys()))
    top_n = st.sidebar.slider("Number of Skills to Show", 10, 100, 50)
    
    full_df = data_dict[selected_prof]
    df_display = full_df.head(top_n)

    # Line Chart: Skill Trajectory
    st.subheader(f"📈 Skill Trajectory: {selected_prof}")
    target_skill = st.selectbox("Analyze Career Path for Skill:", full_df["skills_found"].to_list())
    
    skill_row = full_df.filter(pl.col("skills_found") == target_skill)
    if not skill_row.is_empty():
        plot_df = pd.DataFrame({
            "Level": ["Junior", "Mid-level", "Senior"],
            "Market Share (%)": [
                float(skill_row["Junior"][0]), 
                float(skill_row["Mid-level"][0]), 
                float(skill_row["Senior"][0])
            ]
        })
        fig = px.line(
            plot_df, x="Level", y="Market Share (%)", markers=True, 
            text=[f"{v:.1f}%" for v in plot_df["Market Share (%)"]], 
            title=f"Trend for '{target_skill}'"
        )
        fig.update_traces(textposition="top center", line_color="#007bff", marker_size=10)
        fig.update_layout(yaxis_range=[0, max(plot_df["Market Share (%)"]) * 1.5])
        st.plotly_chart(fig, use_container_width=True)

    # Strategic Insights
    st.divider()
    st.subheader("💡 Strategic Insights")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📉 **Gatekeepers** (Entry-heavy)")
        gatekeepers = full_df.filter(pl.col("Junior_to_Senior_Diff") < -1.5).sort("Junior", descending=True).head(5)
        for s in gatekeepers["skills_found"].to_list(): st.markdown(f"- {s}")

    with col2:
        st.success("📈 **Differentiators** (Senior-heavy)")
        diffs = full_df.filter(pl.col("Junior_to_Senior_Diff") > 0.5).sort("Senior", descending=True).head(5)
        for s in diffs["skills_found"].to_list(): st.markdown(f"- {s}")

    with col3:
        st.warning("🌉 **Bridge Skills** (Mid-level Peak)")
        bridges = full_df.filter(
            (pl.col("Mid-level") > pl.col("Junior")) & 
            (pl.col("Mid-level") > pl.col("Senior"))
        ).sort("Mid-level", descending=True).head(5)
        for s in bridges["skills_found"].to_list(): st.markdown(f"- {s}")

    # Full Data Table
    st.subheader("🔍 Full Evolution Matrix")
    st.dataframe(
        df_display.to_pandas().style.background_gradient(subset=["Total_Rel_Change_Pct"], cmap="RdYlGn"), 
        use_container_width=True
    )

    # --- ADD THIS EXPLANATION BLOCK HERE ---
    with st.expander("📖 How to read this Matrix"):
        st.markdown("""
        This table tracks the **Market Share** of specific skills across different seniority levels.
        
        * **Junior / Mid-level / Senior**: These columns represent the percentage of job postings at that level that require the skill. (Eg. In the 'Senior' category, X% of job postings require this specific skill.)
        * **Junior_to_Senior_Diff**: The absolute change in market share via Senior% - Junior%
            * *Positive (+)*: This skill is more common among Senior job postings. (eg. if 'Cloud computing' has a +30%, it proves that this skill is a requirement to move up the career ladder from Junior to Senior job roles.)
            * *Negative (-)*: This skill is more common among Junior job postings. It's a "Gatekeeper" skill—vital for entry but assumed or less emphasized at senior levels.
        * **Total_Rel_Change_Pct**: The growth rate of the skill's importance as defined by [(Senior-Junior)/Junior]x100. Hence it is calculating relative growth ie. shows how much faster a skill grows in importance compared to where it started. (eg. an absolute difference might be small like 5% but if the Total_Rel_Change_Pct is 400%, it means that the skill is almost exclusively found in Senior job postings.)
            * 🟩 **Green (High %)**: Explosive growth in demand as seniority increases.
            * 🟥 **Red (Negative %)**: Sharp decline in explicit mentions as you move toward Senior roles.
        """)

else:
    st.info("👋 Ready to start. Please upload your dataset in the sidebar and run the analysis.")