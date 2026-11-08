import pandas as pd
import glob
import os

# --- 1. Configuration ---
print("🚀 1. Initializing dataset combination script...")

# Define the pattern to find your dataset files.
# The '*' is a wildcard that matches any characters.
file_pattern = 'variant_stats*.csv'

# Define the name for the final, combined output file.
output_filename = 'all_varient_for_hundred_run.csv'


# --- 2. Find all matching dataset files ---
print(f"🔍 2. Searching for files matching the pattern: '{file_pattern}'")
# glob.glob() finds all pathnames matching a specified pattern
csv_files = glob.glob(file_pattern)

if not csv_files:
    print(f"❌ Error: No files found matching '{file_pattern}'.")
    print("Please make sure your dataset files (e.g., 'evoqem_dataset_4q.csv') are in the same folder as this script.")
    exit()

print(f"✅ Found {len(csv_files)} dataset files to combine:")
for f in csv_files:
    print(f"  - {f}")


# --- 3. Load and combine the files ---
print("\n🔄 3. Loading and combining all files...")
# Create a list to hold all the individual DataFrames
dataframe_list = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        dataframe_list.append(df)
        print(f"  - Loaded '{file}' with {len(df)} rows.")
    except Exception as e:
        print(f"⚠️ Warning: Could not read file '{file}'. Error: {e}")

if not dataframe_list:
    print("❌ Error: No data could be loaded. Aborting.")
    exit()

# Concatenate all the DataFrames in the list into one
combined_df = pd.concat(dataframe_list, ignore_index=True)


# --- 4. Save the combined dataset ---
print("\n💾 4. Saving the final combined dataset...")
try:
    combined_df.to_csv(output_filename, index=False)
    print("\n" + "="*50)
    print("🎉 SUCCESS! All datasets have been combined.")
    print(f"      - Total files combined: {len(csv_files)}")
    print(f"      - Total data points (rows): {len(combined_df)}")
    print(f"      - Final dataset saved as: '{output_filename}'")
    print("="*50)
except Exception as e:
    print(f"❌ Error: Failed to save the combined file. Error: {e}")