import pandas as pd
import time
from transformers import pipeline
import os
import torch
from tqdm import tqdm

# --- 1. CONFIGURATION ---
INPUT_DATA_PATH = './dataset/cleaned_data.csv'
DATE_COLUMN = 'jobOpening_approvedAt'
PROFESSION_COL = 'jobOpening_professionFinal'
EXPERIENCE_COL = 'jobOpening_workExperienceYearsFinal'

START_DATE = '2025-04-01'
END_DATE = '2025-10-01'

# --- 2. ROBUST MODEL SETUP ---
print("Loading NER model...")

# 1. Detect Hardware
try:
    if torch.cuda.is_available():
        device = 0
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = -1
        print("⚠️ No GPU detected. Using CPU (Batch optimized).")

    # 2. Load Pipeline
    skill_extractor = pipeline(
        "ner", 
        model="Nucha/Nucha_ITSkillNER_BERT", 
        aggregation_strategy="simple",
        device=device
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Tip: If using CPU, ensure you have 'torch' installed (pip install torch).")
    exit()

# --- 3. DATA LOADING AND FILTERING ---

try:
    # Added low_memory=False to fix DtypeWarning
    df = pd.read_csv(INPUT_DATA_PATH, low_memory=False)
except FileNotFoundError:
    print(f"Error: '{INPUT_DATA_PATH}' not found.")
    exit()

print(f"\nOriginal job count: {len(df)}")

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

# # --- 4. BATCH PROCESSING (The Fast Way) ---

# print("Running Batch Analysis...")
# start_time = time.time()

# # A generator function to feed text to the model efficiently
# # This prevents memory overflows and allows batching
# def data_generator(dataframe):
#     for text in dataframe['cleaned_text']:
#         # Ensure text is string and truncate to 512 chars to prevent model errors
#         if isinstance(text, str):
#             yield text[:512] 
#         else:
#             yield ""

# results = []
# # Batch size of 32 is a good sweet spot for both CPU and GPU
# BATCH_SIZE = 32

# # The loop where the magic happens
# for output in skill_extractor(data_generator(filtered_df), batch_size=BATCH_SIZE):
#     found_skills = set()
#     for entity in output:
#         # Extract Valid Skills
#         if entity['entity_group'] in ['HSKILL', 'SSKILL']:
#             found_skills.add(entity['word'].strip().lower())
#     results.append(list(found_skills))

# # Assign results back to dataframe
# filtered_df['skills_found'] = results

# end_time = time.time()
# print(f"NER analysis complete. Took {end_time - start_time:.2f} seconds.")













# --- 4. BATCH PROCESSING (The Fast Way) ---

print("Running Batch Analysis...")
start_time = time.time()

# A generator function to feed text to the model efficiently
# This prevents memory overflows and allows batching
def data_generator(dataframe):
    for text in dataframe['cleaned_text']:
        if isinstance(text, str):
            # ⚠️ Manually truncate to 512 words (approx 2000 chars)
            # This avoids the error and keeps the script fast
            yield text[:2000] 
        else:
            yield ""

results = []
BATCH_SIZE = 32
total_rows = len(filtered_df)

print(f"Processing {total_rows} job descriptions...")

# 1. We removed 'truncation=True' from the pipeline call to fix the TypeError.
# 2. We rely on 'text[:2000]' in the generator above to handle length.
for output in tqdm(skill_extractor(data_generator(filtered_df), batch_size=BATCH_SIZE), total=total_rows, unit="job"):
    found_skills = set()
    for entity in output:
        # Extract Valid Skills
        if entity['entity_group'] in ['HSKILL', 'SSKILL']:
            found_skills.add(entity['word'].strip().lower())
    results.append(list(found_skills))

# Assign results back to dataframe
filtered_df['skills_found'] = results

end_time = time.time()
print(f"NER analysis complete. Took {end_time - start_time:.2f} seconds.")





















# --- 5. FREQUENCY ANALYSIS (Raw Counts) ---

print("\n--- 📊 Frequency Analysis ---")

# Create output directory if it doesn't exist
os.makedirs('./output/2d/', exist_ok=True)

# Explode and Count
skills_df = filtered_df.explode('skills_found')
skills_df = skills_df.dropna(subset=['skills_found'])

print(f"Processing skills for ALL professions. Rows: {len(skills_df)}")

skill_counts = skills_df.groupby([
    PROFESSION_COL, 
    EXPERIENCE_COL, 
    'skills_found'
]).size().rename('Raw_Count')

skill_counts.to_csv('./output/2d/2d_frequencyReport.csv')
print(f"✅ Raw frequency report saved.")

# --- 6. STATISTICAL ANALYSIS (Lift Ratio) ---

print("\n--- ✨ Statistical Lift Ratio Analysis ---")

df_counts = skill_counts.reset_index()
total_mentions = df_counts.groupby([PROFESSION_COL, EXPERIENCE_COL])['Raw_Count'].sum()

df_analysis = df_counts.merge(
    total_mentions.rename('Total_Mentions'),
    left_on=[PROFESSION_COL, EXPERIENCE_COL], 
    right_index=True
)

df_analysis['Relative_Freq'] = df_analysis['Raw_Count'] / df_analysis['Total_Mentions']
df_analysis['Prof_Skill_ID'] = df_analysis[PROFESSION_COL] + ' | ' + df_analysis['skills_found']

df_pivot = df_analysis.pivot(
    index='Prof_Skill_ID', 
    columns=EXPERIENCE_COL, 
    values='Relative_Freq'
)

# Calculate Lifts
# Note: This assumes 'Mid-Level', 'Senior', and 'Junior' exist in your data.
# If these specific strings aren't in your Experience column, these columns will be NaN.
if 'Mid-Level' in df_pivot.columns and 'Junior' in df_pivot.columns:
    df_pivot['Lift_Mid_vs_Junior'] = df_pivot['Mid-Level'] / df_pivot['Junior']
if 'Senior' in df_pivot.columns and 'Junior' in df_pivot.columns:
    df_pivot['Lift_Senior_vs_Junior'] = df_pivot['Senior'] / df_pivot['Junior']

df_pivot = df_pivot.replace({float('inf'): 9999})
df_pivot = df_pivot.fillna(0)

df_pivot.to_csv('./output/2d/2d_liftAnalysisReport.csv')
print(f"✅ Lift Ratio Analysis saved.")
print("\n--- 🚀 Master Script Finished ---")