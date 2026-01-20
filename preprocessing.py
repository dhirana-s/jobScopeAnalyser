# import polars as pl
# import os
# import re
# from transformers import pipeline
# from thefuzz import process, fuzz
# from bs4 import BeautifulSoup
# from tqdm import tqdm

# # --- 1. CONFIGURATION ---
# INPUT_PATH = "dataset/cleaned_file.csv"
# OUTPUT_DIR = "dataset"
# BATCH_SIZE = 4 
# FUZZY_THRESHOLD = 92
# TARGET_PROFESSIONS = ['SOFTWARE_ENGINEER', 'AI_ENGINEER']
# PROFESSION_COL = 'jobOpening_professionFinal' 

# # --- 2. CLEANING UTILITIES ---
# def clean_html_to_text(html):
#     if not html: return ""
#     return BeautifulSoup(html, "html.parser").get_text(separator=" ")

# def clean_extracted_token(skill_text):
#     if not skill_text: return None
#     clean = skill_text.replace("##", "").strip().lower()
#     clean = re.sub(r'[^a-z0-9\s\+\#\.]', '', clean)
#     if len(clean) < 2 or clean.isdigit(): return None
#     return " ".join(clean.split())

# # --- 3. LOAD & FILTER ---
# print("--- Step 1: Loading and Filtering Data ---")
# df = pl.read_csv(INPUT_PATH)

# # Rename column if needed
# if "jobOpening_profession" in df.columns:
#     df = df.rename({"jobOpening_profession": PROFESSION_COL})

# # Filter for specific professions
# df = df.filter(pl.col(PROFESSION_COL).is_in(TARGET_PROFESSIONS))
# print(f"Total applications found: {len(df)}")

# # DEDUPLICATION: Count unique job postings, not individual applications
# df = df.unique(subset=["jobOpening_serialNumber"])
# print(f"Unique Job Postings to analyze: {len(df)}")

# # Clean the HTML scope
# df = df.with_columns(
#     pl.col("jobOpening_jobScope").map_elements(clean_html_to_text, return_dtype=pl.String).alias("clean_text")
# )

# # Optimization: Only send unique descriptions to BERT
# unique_texts_df = df.select("clean_text").unique().drop_nulls()
# print(f"Unique descriptions for BERT: {len(unique_texts_df)}")




# # --- 4. BERT EXTRACTION (Full Landscape Version) ---
# print("\n--- Step 2: Initializing BERT NER ---")
# skill_extractor = pipeline(
#     "ner", 
#     model="Nucha/Nucha_ITSkillNER_BERT", 
#     aggregation_strategy="simple",
#     device=-1 
# )

# unique_text_list = unique_texts_df["clean_text"].to_list()
# unique_results = []

# print(f"🚀 Running Full-Text Extraction...")
# for text in tqdm(unique_text_list, desc="Processing Scopes"):
#     if not text:
#         unique_results.append([])
#         continue
    
#     # Logic: Break text into 500-word chunks so BERT reads everything
#     # This prevents the 1,000 character cutoff
#     words = text.split()
#     chunk_size = 300 
#     chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
#     aggregated_skills = set()
#     for chunk in chunks:
#         results = skill_extractor(chunk)
#         for res in results:
#             if res['entity_group'] in ['HSKILL', 'SSKILL']:
#                 skill = clean_extracted_token(res['word'])
#                 if skill:
#                     aggregated_skills.add(skill)
    
#     unique_results.append(list(aggregated_skills))

# # Map skills back
# unique_mapping_df = unique_texts_df.with_columns(pl.Series("raw_skills", unique_results))
# df = df.join(unique_mapping_df, on="clean_text", how="left")




# # --- 5. FUZZY DEDUPLICATION ---
# print("\n--- Step 3: Fuzzy Deduplication ---")
# all_found = df.select(pl.col("raw_skills").flatten()).unique().to_series().to_list()
# all_found = sorted([s for s in all_found if s], key=len)

# mapping = {}
# processed = set()
# for skill in tqdm(all_found, desc="Grouping Similar Skills"):
#     if skill in processed: continue
#     processed.add(skill)
#     mapping[skill] = skill
#     candidates = [c for c in all_found if c not in processed]
#     if not candidates: continue
    
#     match = process.extractOne(skill, candidates, scorer=fuzz.ratio)
#     if match and match[1] >= FUZZY_THRESHOLD:
#         mapping[match[0]] = skill
#         processed.add(match[0])

# # --- 6. SAVE PATTERN MATRIX ---
# print("\n--- Step 4: Saving Results ---")
# final_df = (
#     df.explode("raw_skills")
#     .with_columns(pl.col("raw_skills").replace(mapping).alias("skills_found"))
#     .drop_nulls("skills_found")
# )

# patterns = (
#     final_df.group_by([PROFESSION_COL, "jobOpening_workExperienceYears", "skills_found"])
#     .len() 
#     .sort("len", descending=True)
# )

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# patterns.write_csv(f"{OUTPUT_DIR}/job_scope_patterns.csv")

# print(f"✅ Preprocessing finished! Pattern file created: {OUTPUT_DIR}/job_scope_patterns.csv")





































































# import polars as pl
# import re
# from thefuzz import process, fuzz
# from bs4 import BeautifulSoup

# def clean_html_to_text(html):
#     if not html: return ""
#     return BeautifulSoup(html, "html.parser").get_text(separator=" ")

# def clean_extracted_token(skill_text):
#     if not skill_text: return None
#     clean = skill_text.replace("##", "").strip().lower()
#     clean = re.sub(r'[^a-z0-9\s\+\#\.]', '', clean)
#     if len(clean) < 2 or clean.isdigit(): return None
#     return " ".join(clean.split())

# def run_preprocessing(df, skill_extractor):
#     PROFESSION_COL = 'jobOpening_professionFinal'
#     TARGET_PROFESSIONS = ['SOFTWARE_ENGINEER', 'AI_ENGINEER']

#     # Filter & Deduplicate
#     if "jobOpening_profession" in df.columns:
#         df = df.rename({"jobOpening_profession": PROFESSION_COL})
    
#     df = df.filter(pl.col(PROFESSION_COL).is_in(TARGET_PROFESSIONS))
#     df = df.unique(subset=["jobOpening_serialNumber"])

#     # Clean HTML
#     df = df.with_columns(
#         pl.col("jobOpening_jobScope").map_elements(clean_html_to_text, return_dtype=pl.String).alias("clean_text")
#     )

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
#     mapping = {}
#     processed = set()
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

#     # Return the pattern matrix
#     patterns = (
#         df.explode("raw_skills")
#         .with_columns(pl.col("raw_skills").replace(mapping).alias("skills_found"))
#         .drop_nulls("skills_found")
#         .group_by([PROFESSION_COL, "jobOpening_workExperienceYears", "skills_found"])
#         .len()
#     )
#     return patterns


















































#SNOWFLAKE
import polars as pl
import re
from thefuzz import process, fuzz
from bs4 import BeautifulSoup

def clean_html_to_text(html):
    if not html: return ""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ")

def clean_extracted_token(skill_text):
    if not skill_text: return None
    clean = skill_text.replace("##", "").strip().lower()
    clean = re.sub(r'[^a-z0-9\s\+\#\.]', '', clean)
    if len(clean) < 2 or clean.isdigit(): return None
    return " ".join(clean.split())

def run_preprocessing(df, skill_extractor):
    PROFESSION_COL = 'jobOpening_professionFinal'
    TARGET_PROFESSIONS = ['SOFTWARE_ENGINEER', 'AI_ENGINEER']

    # Rename and Filter
    if PROFESSION_COL not in df.columns:
        df = df.rename({"jobOpening_profession": PROFESSION_COL})
    
    df = df.with_columns(pl.col("jobOpening_jobScope").map_elements(clean_html_to_text, return_dtype=pl.String).alias("clean_text"))

    # BERT Extraction
    unique_texts_df = df.select("clean_text").unique().drop_nulls()
    unique_results = []
    for text in unique_texts_df["clean_text"].to_list():
        words = text.split()
        chunks = [" ".join(words[i:i + 300]) for i in range(0, len(words), 300)]
        aggregated_skills = set()
        for chunk in chunks:
            results = skill_extractor(chunk)
            for res in results:
                if res['entity_group'] in ['HSKILL', 'SSKILL']:
                    skill = clean_extracted_token(res['word'])
                    if skill: aggregated_skills.add(skill)
        unique_results.append(list(aggregated_skills))

    unique_mapping_df = unique_texts_df.with_columns(pl.Series("raw_skills", unique_results))
    df = df.join(unique_mapping_df, on="clean_text", how="left")

    # Fuzzy Deduplication
    all_found = df.select(pl.col("raw_skills").flatten()).unique().to_series().to_list()
    mapping, processed = {}, set()
    for skill in sorted([s for s in all_found if s], key=len):
        if skill in processed: continue
        processed.add(skill)
        mapping[skill] = skill
        candidates = [c for c in all_found if c not in processed]
        if candidates:
            match = process.extractOne(skill, candidates, scorer=fuzz.ratio)
            if match and match[1] >= 92:
                mapping[match[0]] = skill
                processed.add(match[0])

    return df.explode("raw_skills").with_columns(pl.col("raw_skills").replace(mapping).alias("skills_found")).drop_nulls("skills_found")