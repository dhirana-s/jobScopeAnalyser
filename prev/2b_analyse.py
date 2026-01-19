import pandas as pd
import time
from transformers import pipeline
import os

# --- 1. CONFIGURATION ---
# Define the files and columns used in your project
INPUT_DATA_PATH = './dataset/cleaned_data.csv'
DATE_COLUMN = 'jobOpening_approvedAt'
PROFESSION_COL = 'jobOpening_professionFinal'
EXPERIENCE_COL = 'jobOpening_workExperienceYearsFinal'

# Define the date range for filtering (Apr 2025 - Sep 2025)
START_DATE = '2025-04-01'
END_DATE = '2025-10-01' # Up to, but not including, October 1st

# --- 2. BERT MODEL SETUP ---
print("Loading NER model... (This may take a moment on first run)")
try:
    skill_extractor = pipeline(
        "ner", 
        model="Nucha/Nucha_ITSkillNER_BERT", 
        aggregation_strategy="simple"
    )
except ImportError as e:
    print(f"Error loading transformers: {e}. Did you run 'pip install torch'?")
    exit()


def find_skills_transformer(cleaned_text):
    """Uses a BERT model to extract and clean unique skills."""
    if not isinstance(cleaned_text, str) or len(cleaned_text) < 20:
        return []

    # Truncate text (safe for 512-token BERT limit)
    max_length = 1500
    truncated_text = cleaned_text[:max_length]

    try:
        skills = skill_extractor(truncated_text)
        found_skills = set()
        for entity in skills:
            if entity['entity_group'] in ['HSKILL', 'SSKILL']:
                # Clean up the word for consistent statistical grouping
                skill_word = entity['word'].strip().lower()
                found_skills.add(skill_word)
        return list(found_skills)
    except Exception:
        return []

# --- 3. DATA LOADING AND FILTERING ---

try:
    df = pd.read_csv(INPUT_DATA_PATH)
except FileNotFoundError:
    print(f"Error: '{INPUT_DATA_PATH}' not found.")
    exit()

print(f"\nOriginal job count: {len(df)}")
print("Applying date filter and running BERT analysis...")

# Date Conversion and Filtering
if DATE_COLUMN in df.columns:
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    df = df.dropna(subset=[DATE_COLUMN])
    date_filter = (df[DATE_COLUMN] >= START_DATE) & (df[DATE_COLUMN] < END_DATE)
    filtered_df = df[date_filter].copy()
    print(f"Filtered job count ({START_DATE} to {END_DATE}): {len(filtered_df)}")
else:
    print(f"Warning: Date column '{DATE_COLUMN}' not found. Using all data.")
    filtered_df = df.copy()

# Apply the BERT model
start_time = time.time()
filtered_df['skills_found'] = filtered_df['cleaned_text'].apply(find_skills_transformer)
end_time = time.time()
print(f"NER analysis complete. Took {end_time - start_time:.2f} seconds.")


# --- 4. FREQUENCY ANALYSIS (Raw Counts) ---

print("\n--- 📊 Frequency Analysis ---")

# Explode the list of skills into separate rows
skills_df = filtered_df.explode('skills_found')
skills_df = skills_df.dropna(subset=['skills_found'])

# Calculate the raw count (Count)
skill_counts = skills_df.groupby([
    PROFESSION_COL, 
    EXPERIENCE_COL, 
    'skills_found'
]).size().rename('Raw_Count')

# Save the raw count report
skill_counts.to_csv('skill_frequency_report_SWE.csv')
print(f"✅ Raw frequency report saved.")







# --- 5. STATISTICAL ANALYSIS (Lift Ratio) ---

print("\n--- ✨ Statistical Lift Ratio Analysis ---")

# 1. Convert Series to DataFrame for analysis
# NOTE: The raw counts (Raw_Count) are already correctly grouped by all three variables
df_counts = skill_counts.reset_index()

# 2. Calculate Total Skill Mentions (TL) per level
# This is unchanged and correct
total_mentions = df_counts.groupby(EXPERIENCE_COL)['Raw_Count'].sum()

# 3. Merge Total Mentions back into the DataFrame
df_analysis = df_counts.merge(
    total_mentions.rename('Total_Mentions'),
    left_on=EXPERIENCE_COL,
    right_index=True
)

# 4. Calculate Relative Frequency (Ps|L) - Normalization
df_analysis['Relative_Freq'] = df_analysis['Raw_Count'] / df_analysis['Total_Mentions']

# 5. Pivot for Ratio Calculation
# ⭐️ THE FIX: Combine Profession and Skill into a new index before pivoting ⭐️

# 5a. Create a unique identifier combining Profession and Skill
df_analysis['Prof_Skill_ID'] = df_analysis[PROFESSION_COL] + ' | ' + df_analysis['skills_found']

# 5b. Pivot using the new unique ID as the index
df_pivot = df_analysis.pivot(
    index='Prof_Skill_ID',  # Unique combo of Profession and Skill
    columns=EXPERIENCE_COL, 
    values='Relative_Freq'
)

# 6. Calculate Lift Ratios (Metric 2)
df_pivot['Lift_Mid_vs_Junior'] = df_pivot['Mid-Level'] / df_pivot['Junior']
df_pivot['Lift_Senior_vs_Junior'] = df_pivot['Senior'] / df_pivot['Junior']

# Clean up infinite/NaN values for presentation
df_pivot = df_pivot.replace({float('inf'): 9999})
df_pivot = df_pivot.fillna(0)

# Save the final analysis table
df_pivot.to_csv('skill_lift_ratio_analysis.csv')
print(f"✅ Lift Ratio Analysis saved.")