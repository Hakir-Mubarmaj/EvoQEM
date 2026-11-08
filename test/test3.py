import pandas as pd

# ফাইল এর নাম
input_file = 'test_fidelity_analysis.csv'
output_file = 'test_fidelity_analysis_true_only.csv'

# ১. ফাইল লোড করা
df = pd.read_csv(input_file)

# ২. ফিল্টার করা: যেখানে is_cross_match == True শুধু সেগুলো রাখা
# (এর মানেই হলো False গুলো ডিলিট হয়ে যাবে)
df_filtered = df[df['is_cross_match'] == True]

# ৩. সেভ করা
df_filtered.to_csv(output_file, index=False)

print(f"কাজ শেষ! False গুলো ডিলিট করে {output_file} এ সেভ করা হয়েছে।")
print(f"মোট ডাটা ছিল: {len(df)}, এখন আছে: {len(df_filtered)}")