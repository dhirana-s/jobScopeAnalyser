import polars as pl
import os

# 1. Define Paths 
# Since dtcExtract.xlsx is in the SAME folder as this script, just use the name!
input_path = "dtcExtract.xlsx" 
output_dir = "dataset"
output_file = os.path.join(output_dir, "cleaned_file.csv")

# 2. Load the specific sheet using 'calamine'
df = pl.read_excel(
    input_path, 
    sheet_name="jobApplications_FY24",
    engine="calamine"
)

# 3. Apply Cleaning Logic
cleaned_df = (
    df.with_columns(
        # Standardize jobOpening_profession
        pl.col("jobOpening_profession")
        .str.replace(r"SOFTWARE_ENGINEER_.*", "SOFTWARE_ENGINEER"),
        
        # Map experience levels
        pl.col("jobOpening_workExperienceYears").replace({
            "NO_EXPERIENCE": "Junior",
            "ONE_TWO_YEARS": "Junior",
            "THREE_FIVE_YEARS": "Mid-level",
            "SIX_TEN_YEARS": "Senior",
            "GREATER_TEN_YEARS": "Senior"
        })
    )
)

# 4. Save to CSV
os.makedirs(output_dir, exist_ok=True)
cleaned_df.write_csv(output_file)

print(f"Success! Cleaned file saved to: {output_file}")

















# SNOWFLAKE

import polars as pl
import pandas as pd

def run_cleaning(uploaded_file, date_range=None):
    # Load data
    df = pl.read_excel(uploaded_file, sheet_name="jobApplications_FY24", engine="calamine")
    
    # 1. Date Filtering Logic
    df = df.with_columns(
        pl.col("jobOpening_submittedAt").cast(pl.Datetime).alias("submitted_at_dt")
    )
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        df = df.filter(
            pl.col("submitted_at_dt").is_between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        )

    # 2. Standard Cleaning
    cleaned_df = df.with_columns(
        pl.col("jobOpening_profession").str.replace(r"SOFTWARE_ENGINEER_.*", "SOFTWARE_ENGINEER"),
        pl.col("jobOpening_workExperienceYears").replace({
            "NO_EXPERIENCE": "Junior", "ONE_TWO_YEARS": "Junior",
            "THREE_FIVE_YEARS": "Mid-level", "SIX_TEN_YEARS": "Senior", "GREATER_TEN_YEARS": "Senior"
        })
    )
    return cleaned_df