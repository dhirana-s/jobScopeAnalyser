import pandas as pd
import re  
import html 

def clean_text_with_regex(raw_text):
    if not isinstance(raw_text, str):
        return ""

    # 1. Un-escape HTML entities (e.g., &nbsp; -> ' ', &lt; -> '<')
    text = html.unescape(raw_text)

    # 2. Define the regex pattern to find HTML tags
    # r'<.*?>' finds anything that starts with <, ends with >, and has any characters in between (non-greedy)
    tag_re = re.compile(r'<.*?>')

    # 3. Replace all found tags with a single space
    text = re.sub(tag_re, ' ', text)

    # 4. Normalize all whitespace (newlines, tabs, multiple spaces)
    # to a single space, convert to lowercase, and strip ends.
    text = re.sub(r'\s+', ' ', text).strip().lower()

    return text




# --- MAIN SCRIPT ---

# 1. Define your file names
excel_file_name = './dataset/raw.xlsx'
clean_output_file = './dataset/cleaned_data.csv'

# 2. Find the name of your messy column
try:
    preview_df = pd.read_excel(excel_file_name, nrows=5)
except FileNotFoundError:
    print(f"Error: File not found. Make sure '{excel_file_name}' is in the same folder as this script.")
    exit()

print("Found these columns in your file:")
print(preview_df.columns.to_list())

# --- IMPORTANT: UPDATE THIS BASED ON THE OUTPUT ABOVE ---
messy_column_name = 'jobOpening_jobScope' # <--- CHECK AND UPDATE THIS

# 3. Load the full Excel file
print("\nLoading full Excel file...")
df = pd.read_excel(excel_file_name)

# 4. Apply the NEW cleaning function to the entire column
print(f"Cleaning text from column: '{messy_column_name}'...")
df['cleaned_text'] = df[messy_column_name].apply(clean_text_with_regex)

# 5. Save your clean data
df.to_csv(clean_output_file, index=False)

print("\n--- 🥳 Success! (Third time's the charm) ---")
print(f"Clean data saved to: {clean_output_file}")

# 6. Print a preview
# Get the first two columns (e.g., 'Role', 'level') + our new clean column
try:
    cols_to_print = df.columns[0:2].to_list() + ['cleaned_text']
    print(df[cols_to_print].head())
except Exception as e:
    print(f"\nCould not print preview (error: {e}), but file is saved.")
    print("Check the 'cleaned_job_scopes.csv' file!")