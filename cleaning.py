# import polars as pl
# import os

# # 1. Define Paths 
# # Since dtcExtract.xlsx is in the SAME folder as this script, just use the name!
# input_path = "dtcExtract.xlsx" 
# output_dir = "dataset"
# output_file = os.path.join(output_dir, "cleaned_file.csv")

# # 2. Load the specific sheet using 'calamine'
# df = pl.read_excel(
#     input_path, 
#     sheet_name="jobApplications_FY24",
#     engine="calamine"
# )

# # 3. Apply Cleaning Logic
# cleaned_df = (
#     df.with_columns(
#         # Standardize jobOpening_profession
#         pl.col("jobOpening_profession")
#         .str.replace(r"SOFTWARE_ENGINEER_.*", "SOFTWARE_ENGINEER"),
        
#         # Map experience levels
#         pl.col("jobOpening_workExperienceYears").replace({
#             "NO_EXPERIENCE": "Junior",
#             "ONE_TWO_YEARS": "Junior",
#             "THREE_FIVE_YEARS": "Mid-level",
#             "SIX_TEN_YEARS": "Senior",
#             "GREATER_TEN_YEARS": "Senior"
#         })
#     )
# )

# # 4. Save to CSV
# os.makedirs(output_dir, exist_ok=True)
# cleaned_df.write_csv(output_file)

# print(f"Success! Cleaned file saved to: {output_file}")

















# # SNOWFLAKE

# import polars as pl
# import pandas as pd

# def run_cleaning(uploaded_file, date_range=None):
#     # Load data
#     df = pl.read_excel(uploaded_file, sheet_name="jobApplications_FY24", engine="calamine")
    
#     # 1. Flexible Date Filtering Logic
#     # Check if the column is already datetime or needs conversion
#     if df["jobOpening_submittedAt"].dtype == pl.String:
#         df = df.with_columns(
#             pl.col("jobOpening_submittedAt")
#             .str.to_date("%m/%d/%Y")
#             .cast(pl.Datetime)
#             .alias("submitted_at_dt")
#         )
#     else:
#         # If already a date/datetime, just ensure it's the right format for filtering
#         df = df.with_columns(
#             pl.col("jobOpening_submittedAt").cast(pl.Datetime).alias("submitted_at_dt")
#         )

#     # Filtering by date range
#     if date_range and len(date_range) == 2:
#         start_date, end_date = date_range
#         df = df.filter(
#             pl.col("submitted_at_dt").is_between(pd.Timestamp(start_date), pd.Timestamp(end_date))
#         )

#     # 2. Standard Cleaning (Profession & Experience)
#     cleaned_df = df.with_columns(
#         pl.col("jobOpening_profession").str.replace(r"SOFTWARE_ENGINEER_.*", "SOFTWARE_ENGINEER"),
#         pl.col("jobOpening_workExperienceYears").replace({
#             "NO_EXPERIENCE": "Junior", 
#             "ONE_TWO_YEARS": "Junior",
#             "THREE_FIVE_YEARS": "Mid-level", 
#             "SIX_TEN_YEARS": "Senior", 
#             "GREATER_TEN_YEARS": "Senior"
#         }, default=pl.col("jobOpening_workExperienceYears")) # Added default to prevent nulls
#     )
    
#     return cleaned_df





































# # PURPLE
# import polars as pl
# import pandas as pd

# def run_cleaning(uploaded_file, date_range=None):
#     # Load data
#     df = pl.read_excel(uploaded_file, sheet_name="jobApplications_FY24", engine="calamine")
    
#     # 1. Profession Normalization (Do this BEFORE unique check)
#     df = df.with_columns(
#         pl.col("jobOpening_profession").str.replace(r"SOFTWARE_ENGINEER_.*", "SOFTWARE_ENGINEER").alias("jobOpening_professionFinal")
#     )

#     # 2. THE FIX: Define Unique Job Postings HERE
#     # Only keep the professions we care about and ensure one row per serial number
#     TARGET_PROFESSIONS = ['SOFTWARE_ENGINEER', 'AI_ENGINEER']
#     df = df.filter(
#         pl.col("jobOpening_professionFinal").is_in(TARGET_PROFESSIONS)
#     ).unique(subset=["jobOpening_serialNumber"])

#     # 3. Date Filtering
#     if df["jobOpening_submittedAt"].dtype == pl.String:
#         df = df.with_columns(pl.col("jobOpening_submittedAt").str.to_date("%m/%d/%Y").cast(pl.Datetime).alias("submitted_at_dt"))
#     else:
#         df = df.with_columns(pl.col("jobOpening_submittedAt").cast(pl.Datetime).alias("submitted_at_dt"))

#     if date_range and len(date_range) == 2:
#         start_date, end_date = date_range
#         df = df.filter(pl.col("submitted_at_dt").is_between(pd.Timestamp(start_date), pd.Timestamp(end_date)))

#     # 4. Experience Mapping
#     cleaned_df = df.with_columns(
#         pl.col("jobOpening_workExperienceYears").replace({
#             "NO_EXPERIENCE": "Junior", "ONE_TWO_YEARS": "Junior",
#             "THREE_FIVE_YEARS": "Mid-level", "SIX_TEN_YEARS": "Senior", 
#             "GREATER_TEN_YEARS": "Senior"
#         }, default=pl.col("jobOpening_workExperienceYears"))
#     )

#     # NEW: Capture the absolute truth before BERT processing
#     true_totals = (
#         cleaned_df.group_by(["jobOpening_professionFinal", "jobOpening_workExperienceYears"])
#         .agg(pl.len().alias("total_unique_jobs"))
#     )
    
#     return cleaned_df, true_totals








































































# SHIFTED WORKBOOOKS# PURPLE
import polars as pl
import pandas as pd

def run_cleaning(uploaded_file, date_range=None):
    # Load data
    df = pl.read_excel(uploaded_file, sheet_name="jobOpenings", engine="calamine")
    
    # 1. Profession Normalization
    df = df.with_columns(
        pl.col("jobOpening_profession")
        .str.replace(r"SOFTWARE_ENGINEER_.*", "SOFTWARE_ENGINEER")
        .alias("jobOpening_professionFinal")
    )

    # 2. ADDED FILTERS: Status and Experience Type
    # This keeps only APPROVED/CLOSED and FULL_TIME/CONTRACT
    df = df.filter(
        (pl.col("jobOpening_status").is_in(["APPROVED", "CLOSED"])) &
        (pl.col("jobOpening_experienceType").is_in(["FULL_TIME", "CONTRACT"]))
    )

    # 3. Define Unique Job Postings
    df = df.unique(subset=["jobOpening_serialNumber"])

    # 4. Date Filtering
    if df["jobOpening_submittedAt"].dtype == pl.String:
        df = df.with_columns(pl.col("jobOpening_submittedAt").str.to_date("%m/%d/%Y").cast(pl.Datetime).alias("submitted_at_dt"))
    else:
        df = df.with_columns(pl.col("jobOpening_submittedAt").cast(pl.Datetime).alias("submitted_at_dt"))

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        df = df.filter(pl.col("submitted_at_dt").is_between(pd.Timestamp(start_date), pd.Timestamp(end_date)))

    # 5. Experience Mapping
    cleaned_df = df.with_columns(
        pl.col("jobOpening_workExperienceYears").replace({
            "NO_EXPERIENCE": "Junior", "ONE_TWO_YEARS": "Junior",
            "THREE_FIVE_YEARS": "Mid-level", "SIX_TEN_YEARS": "Senior", 
            "GREATER_TEN_YEARS": "Senior"
        }, default=pl.col("jobOpening_workExperienceYears"))
    )

    # Capture totals
    true_totals = (
        cleaned_df.group_by(["jobOpening_professionFinal", "jobOpening_workExperienceYears"])
        .agg(pl.len().alias("total_unique_jobs"))
    )
    
    return cleaned_df, true_totals