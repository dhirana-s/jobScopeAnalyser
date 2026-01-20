# import polars as pl
# import os

# # --- 1. CONFIGURATION ---
# INPUT_PATH = "dataset/job_scope_patterns.csv"
# OUTPUT_DIR = "reports"
# PROFESSION_COL = 'jobOpening_professionFinal'

# # Combined and deduplicated STOP_WORDS list
# STOP_WORDS = [
#     "and the", "with", "are", "er", "less", "inform", "aw", "and", "the", "of", 
#     "to", "in", "for", "on", "at", "by", "from", "as", "with a", "and an", "full", 
#     "natural", "technical", "motivated", "command", "computing and", 
#     "solving skills and", "skills", "attention", "organization", "is", "be", "an", "red"
# ]

# if not os.path.exists(INPUT_PATH):
#     print(f"Error: {INPUT_PATH} not found. Please run 2_preprocessing.py first!")
#     exit()

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# df = pl.read_csv(INPUT_PATH)

# # --- 2. CLEANING & RE-AGGREGATING ---
# # Standardizing fragments and categories across all levels
# df = df.with_columns(
#     pl.col("skills_found").replace({
#         "bachelor": "bachelor's degree",
#         "s degree": "bachelor's degree",
#         "bachelor s": "bachelor's degree",
#         "master": "master's degree",
#         "master s": "master's degree",
#         "apis": "api", 
#         "c + +": "c++", 
#         "c +": "c++",
#         "react. js": "react", 
#         "reactjs": "react",
#         "robotic": "robotics",
#         "machine vision": "computer vision",
#         "verbal communication": "communication",
#         "effective communication": "communication",
#         "communication skills": "communication"
#     })
# ).filter(~pl.col("skills_found").is_in(STOP_WORDS))

# # Re-group to merge rows like 'react' and 'reactjs' before calculating percentages
# df = df.group_by([PROFESSION_COL, "jobOpening_workExperienceYears", "skills_found"]).agg(
#     pl.col("len").sum()
# )

# # --- 3. NORMALIZE BY CATEGORY VOLUME ---
# # This ensures percentages are relative to the total skills found in THAT seniority level
# category_totals = (
#     df.group_by([PROFESSION_COL, "jobOpening_workExperienceYears"])
#     .agg(pl.sum("len").alias("total_mentions_at_level"))
# )

# df_norm = df.join(category_totals, on=[PROFESSION_COL, "jobOpening_workExperienceYears"])
# df_norm = df_norm.with_columns(
#     (pl.col("len") / pl.col("total_mentions_at_level") * 100).alias("share_pct")
# )

# # --- 4. THE EVOLUTION ANALYSIS (Full 3-Level Pipeline) ---
# def get_career_evolution(prof):
#     # Defining the stages to look for
#     levels = ["Junior", "Mid-level", "Senior"] 
    
#     # Filter for the specific profession and the three career stages
#     js_df = df_norm.filter(
#         (pl.col(PROFESSION_COL) == prof) & 
#         (pl.col("jobOpening_workExperienceYears").is_in(levels))
#     )
    
#     # Pivot to create columns for each seniority level
#     pivot = js_df.pivot(
#         values="share_pct",
#         index="skills_found",
#         on="jobOpening_workExperienceYears",
#         aggregate_function="sum"
#     ).fill_null(0)

#     # Safety: Ensure all level columns exist even if data is missing for one
#     for col in levels:
#         if col not in pivot.columns:
#             pivot = pivot.with_columns(pl.lit(0.0).alias(col))

#     # Calculate Career Progression Differentials
#     pivot = pivot.with_columns(
#         (pl.col("Mid-level") - pl.col("Junior")).alias("Junior_to_Mid_Diff"),
#         (pl.col("Senior") - pl.col("Mid-level")).alias("Mid_to_Senior_Diff"),
#         # Total Relative Growth from Junior to Senior
#         pl.when(pl.col("Junior") == 0).then(pl.lit(100.0))
#           .otherwise(((pl.col("Senior") - pl.col("Junior")) / pl.col("Junior")) * 100).alias("Total_Rel_Change_Pct")
#     )
    
#     # Sort by top Junior skills to see how the 'entry requirements' evolve over time
#     return pivot.sort("Junior", descending=True).head(50)

# # Generate Reports
# swe_evolution = get_career_evolution("SOFTWARE_ENGINEER")
# ai_evolution = get_career_evolution("AI_ENGINEER")

# # --- 5. EXPORT ---
# swe_evolution.write_csv(f"{OUTPUT_DIR}/swe_skill_evolution_full.csv")
# ai_evolution.write_csv(f"{OUTPUT_DIR}/ai_skill_evolution_full.csv")

# print("\n" + "="*50)
# print("✅ ANALYSIS SUCCESSFUL: 3-LEVEL CAREER PATH CREATED")
# print(f"Reports saved to: {OUTPUT_DIR}/")
# print("-" * 50)
# print("Columns explained:")
# print("1. Junior/Mid-level/Senior: Market share % of that skill at that level.")
# print("2. Junior_to_Mid_Diff: Increase/Decrease in importance as one moves to Mid.")
# print("3. Total_Rel_Change_Pct: The 'Rise or Fall' percentage over the whole career.")
# print("="*50)


























































# import polars as pl
# import os

# # --- 1. CONFIGURATION ---
# INPUT_PATH = "dataset/job_scope_patterns.csv"
# OUTPUT_DIR = "reports"
# PROFESSION_COL = 'jobOpening_professionFinal'

# # Comprehensive STOP_WORDS list to remove all identified noise/fragments
# STOP_WORDS = [
#     "and the", "with", "are", "er", "less", "inform", "aw", "and", "the", "of", 
#     "to", "in", "for", "on", "at", "by", "from", "as", "with a", "and an", "full", 
#     "natural", "technical", "motivated", "command", "computing and", 
#     "solving skills and", "skills", "attention", "organization", "is", "be", "an",
#     "plus", "red", "deep", "native", "experience" 
# ]

# if not os.path.exists(INPUT_PATH):
#     print(f"Error: {INPUT_PATH} not found. Please run 2_preprocessing.py first!")
#     exit()

# os.makedirs(OUTPUT_DIR, exist_ok=True)
# df = pl.read_csv(INPUT_PATH)

# # --- 2. CLEANING & RE-AGGREGATING ---
# df = df.with_columns(
#     pl.col("skills_found").replace({
#         "bachelor": "bachelor's degree",
#         "s degree": "bachelor's degree",
#         "bachelor s": "bachelor's degree",
#         "master": "master's degree",
#         "master s": "master's degree",
#         "master s degree": "master's degree",
#         "apis": "api", 
#         "c + +": "c++", 
#         "c +": "c++",
#         "react. js": "react", 
#         "reactjs": "react",
#         "robotic": "robotics",
#         "machine vision": "computer vision",
#         "verbal communication": "communication",
#         "effective communication": "communication",
#         "communication skills": "communication",
#         "ml": "machine learning",
#         "gging": "debugging",
#         "native": "react native"
#     })
# ).filter(~pl.col("skills_found").is_in(STOP_WORDS))

# # Merge cleaned rows (aggregating counts for standardized names)
# df = df.group_by([PROFESSION_COL, "jobOpening_workExperienceYears", "skills_found"]).agg(
#     pl.col("len").sum()
# )

# # --- 3. NORMALIZE BY CATEGORY VOLUME ---
# category_totals = (
#     df.group_by([PROFESSION_COL, "jobOpening_workExperienceYears"])
#     .agg(pl.sum("len").alias("total_mentions_at_level"))
# )

# df_norm = df.join(category_totals, on=[PROFESSION_COL, "jobOpening_workExperienceYears"])
# df_norm = df_norm.with_columns(
#     (pl.col("len") / pl.col("total_mentions_at_level") * 100).alias("share_pct")
# )

# # --- 4. THE EVOLUTION ANALYSIS (Full Pipeline) ---
# def get_career_evolution(prof):
#     levels = ["Junior", "Mid-level", "Senior"] 
    
#     js_df = df_norm.filter(
#         (pl.col(PROFESSION_COL) == prof) & 
#         (pl.col("jobOpening_workExperienceYears").is_in(levels))
#     )
    
#     # Pivot for comparison
#     pivot = js_df.pivot(
#         values="share_pct",
#         index="skills_found",
#         on="jobOpening_workExperienceYears",
#         aggregate_function="sum"
#     ).fill_null(0)

#     # Ensure all columns exist
#     for col in levels:
#         if col not in pivot.columns:
#             pivot = pivot.with_columns(pl.lit(0.0).alias(col))

#     # Calculate absolute differentials and relative change
#     pivot = pivot.with_columns(
#         (pl.col("Mid-level") - pl.col("Junior")).alias("Junior_to_Mid_Diff"),
#         (pl.col("Senior") - pl.col("Mid-level")).alias("Mid_to_Senior_Diff"),
#         (pl.col("Senior") - pl.col("Junior")).alias("Junior_to_Senior_Diff"),
#         pl.when(pl.col("Junior") == 0).then(pl.lit(100.0))
#           .otherwise(((pl.col("Senior") - pl.col("Junior")) / pl.col("Junior")) * 100).alias("Total_Rel_Change_Pct")
#     )
    
#     # Explicitly arrange columns in chronological order
#     pivot = pivot.select([
#         "skills_found", "Junior", "Mid-level", "Senior", 
#         "Junior_to_Mid_Diff", "Mid_to_Senior_Diff", "Junior_to_Senior_Diff", "Total_Rel_Change_Pct"
#     ])
    
#     # Increased to top 100 skills
#     return pivot.sort("Junior", descending=True).head(100)


# # Generate Reports
# swe_evolution = get_career_evolution("SOFTWARE_ENGINEER")
# ai_evolution = get_career_evolution("AI_ENGINEER")

# # --- 5. EXPORT ---
# swe_evolution.write_csv(f"{OUTPUT_DIR}/swe_skill_evolution_full.csv")
# ai_evolution.write_csv(f"{OUTPUT_DIR}/ai_skill_evolution_full.csv")


# print("\n" + "="*50)
# print("✅ ANALYSIS SUCCESSFUL: 100 SKILLS PROCESSED")
# print(f"Reports saved to: {OUTPUT_DIR}/")
# print("-" * 50)
# print("Order: Junior -> Mid-level -> Senior")
# print("New Column: Junior_to_Senior_Diff (Absolute net change)")
# print("="*50)






























































#SNOWFLAKE
import polars as pl


STOP_WORDS = [
    "and the", "with", "are", "er", "less", "inform", "aw", "and", "the", "of", 
    "to", "in", "for", "on", "at", "by", "from", "as", "with a", "and an", "full", 
    "natural", "technical", "motivated", "command", "computing and", 
    "solving skills and", "skills", "attention", "organization", "is", "be", "an", "red"
]

def run_analysis(df, true_totals):
    PROFESSION_COL = 'jobOpening_professionFinal'
    
    # 1. Standardization
    df = df.with_columns(
        pl.col("skills_found").replace(
        {
        "bachelor": "bachelor's degree",
        "s degree": "bachelor's degree",
        "bachelor s": "bachelor's degree",
        "bachelor s degree": "bachelor's degree",
        "master": "master's degree",
        "master s": "master's degree",
        "master s degree": "master's degree",
        "apis": "api", 
        "c + +": "c++", 
        "c +": "c++",
        "react. js": "react", 
        "reactjs": "react",
        "node. js": "node.js",
        ". net": ".net",
        "robotic": "robotics",
        "machine vision": "computer vision",
        "verbal communication": "communication",
        "effective communication": "communication",
        "communication skills": "communication",
        "ml": "machine learning",
        "gging": "debugging",
        "native": "react native"
    })
    ).filter(~pl.col("skills_found").is_in(STOP_WORDS))



    # NEW FIXITYYYY

    # # 2a. Calculate THE TRUE DENOMINATOR (Unique Job Postings per Level)
    # # We look at the original df before exploding to find how many unique jobs exist
    # unique_jobs_per_level = (
    #     df.group_by([PROFESSION_COL, "jobOpening_workExperienceYears"])
    #     .agg(pl.col("jobOpening_serialNumber").n_unique().alias("total_unique_jobs"))
    # )

    # 2b. Calculate THE NUMERATOR (How many unique jobs have this specific skill?)
    patterns = (
        df.group_by([PROFESSION_COL, "jobOpening_workExperienceYears", "skills_found"])
        .agg(pl.col("jobOpening_serialNumber").n_unique().alias("jobs_with_skill"))
    )

    # 3. Join with the GOLDEN DENOMINATOR from cleaning.py
    df_norm = patterns.join(
        true_totals, 
        left_on=[PROFESSION_COL, "jobOpening_workExperienceYears"],
        right_on=["jobOpening_professionFinal", "jobOpening_workExperienceYears"],
        how="left"
    ).with_columns(
        (pl.col("jobs_with_skill") / pl.col("total_unique_jobs") * 100).alias("share_pct")
    )

    # # 2c. Join and calculate Share % based on unique jobs
    # df_norm = patterns.join(
    #     unique_jobs_per_level, 
    #     on=[PROFESSION_COL, "jobOpening_workExperienceYears"], 
    #     how="left"
    # ).with_columns(
    #     (pl.col("jobs_with_skill") / pl.col("total_unique_jobs") * 100).alias("share_pct")
    # )

    # END OF NEW FIXITYY


    # # 2. Aggregation & Normalization
    # patterns = df.group_by([PROFESSION_COL, "jobOpening_workExperienceYears", "skills_found"]).len()
    # totals = patterns.group_by([PROFESSION_COL, "jobOpening_workExperienceYears"]).agg(pl.sum("len").alias("total"))
    # df_norm = patterns.join(totals, on=[PROFESSION_COL, "jobOpening_workExperienceYears"])
    # df_norm = df_norm.with_columns((pl.col("len") / pl.col("total") * 100).alias("share_pct"))

    # 3. Evolution Helper
    def get_ev(prof):
        levels = ["Junior", "Mid-level", "Senior"]
        pivot = df_norm.filter(pl.col(PROFESSION_COL) == prof).pivot(
            values="share_pct", index="skills_found", on="jobOpening_workExperienceYears", aggregate_function="sum"
        ).fill_null(0)
        
        for col in levels:
            if col not in pivot.columns: pivot = pivot.with_columns(pl.lit(0.0).alias(col))
            
        pivot = pivot.with_columns(
            (pl.col("Senior") - pl.col("Junior")).alias("Junior_to_Senior_Diff"),
            pl.when(pl.col("Junior") == 0).then(100.0).otherwise(((pl.col("Senior")-pl.col("Junior"))/pl.col("Junior"))*100).alias("Total_Rel_Change_Pct")
        )
        # Add the other diffs here as per your original code
        return pivot.sort("Junior", descending=True).head(100)

    return {"SOFTWARE_ENGINEER": get_ev("SOFTWARE_ENGINEER"), "AI_ENGINEER": get_ev("AI_ENGINEER")}