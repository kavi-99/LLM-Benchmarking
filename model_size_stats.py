# TMR, Coefficient of Variation

import pandas as pd
import glob
import os
import ast

graph_dir = "experiments/zexp_5.2_modelsize_plots/results/results"

exp_dirs = {
    "azure_8b":       "experiments/exp_20250618_164345_f0bbdb78",
    "azure_70b":       "experiments/exp_20250618_171105_1492bb31",
    "azure_671b":       "experiments/exp_20250618_172700_afc8560c",
    "aws_8b":       "experiments/exp_20250618_164129_04a19ed7",
    "aws_70b":       "experiments/exp_20250618_171056_ae2f18a5",
    "aws_671b":       "experiments/exp_20250618_172744_195ee80d",
    "tgt_8b":       "experiments/exp_20250618_184735_750e2005",
    "tgt_70b":       "experiments/exp_20250618_191558_a419f85d",
    "tgt_671b":       "experiments/exp_20250619_202834_35e81537",
    "vllm_8b":       "experiments/exp_20250618_165035_d0f540aa",
    "vllm_70b":       "experiments/exp_20250618_171748_8f758800",
    "vllm_671b":       "experiments/exp_20250619_111919_5ccb399a"
}

# display_names = {
#     'common-model-small': 'LLama 3.1 8B',
#     'common-model': 'LLama 3.3 70B',
#     'deepseek-r1': 'Deepseek R1 671B',
#     'meta-llama/Llama-3.1-8B-Instruct': 'LLama 3.1 8B',
#     '/dataset/shared_models/llama3_3-70b': 'LLama 3.3 70B',
#     '/dataset/shared_models/deepseek-r1': 'Deepseek R1 671B'
# }

display_names = {
    'common-model-small': '8B',
    'common-model': '70B',
    'deepseek-r1': '671B',
    'meta-llama/Llama-3.1-8B-Instruct': '8B',
    '/dataset/shared_models/llama3_3-70b': '70B',
    '/dataset/shared_models/deepseek-r1': '671B'
}


# Debug: Add more detailed printing
print("=== DEBUGGING PLOT ISSUES ===")

# 2) Read each folder's CSVs into a dict of lists-of-DataFrames
dfs = {}
all_dfs = {}
skip_idxs = {0, 3, 7}

for name, folder in exp_dirs.items():
    pattern = os.path.join(folder, "*.csv")
    paths   = glob.glob(pattern)
    print(f"\n{name}: Found {len(paths)} CSV files in {folder}")
    
    all_dfs[name] = [pd.read_csv(p) for p in paths]
    dfs[name] = [
        df
        for idx, df in enumerate(all_dfs[name])
        if idx not in skip_idxs
    ]
    
    print(f"{name}: After filtering, have {len(dfs[name])} DataFrames")
    for i, df in enumerate(dfs[name]):
        print(f"  DataFrame {i}: metric={df['metric'].unique()}, shape={df.shape}")
        # Check for empty dataframes
        if df.empty:
            print(f"    WARNING: DataFrame {i} is empty!")

# 3) Build dfs_all by zipping in the same provider-order
provider_order = list(exp_dirs.keys())
list_of_df_lists = [dfs[p] for p in provider_order]

# Check if all providers have the same number of files
lengths = [len(df_list) for df_list in list_of_df_lists]
print(f"\nProvider DataFrame counts: {dict(zip(provider_order, lengths))}")
if len(set(lengths)) > 1:
    print("WARNING: Not all providers have the same number of DataFrames!")

# 4) Concatenate the i-th file of each provider across all providers
dfs_all = []
for i, dfs_tuple in enumerate(zip(*list_of_df_lists)):
    print(f"\nCombining DataFrame set {i}:")
    for j, df in enumerate(dfs_tuple):
        print(f"  Provider {provider_order[j]}: {df.shape}, metric={df['metric'].unique()}")
    
    combined_df = pd.concat(dfs_tuple, ignore_index=True)
    combined_df['model'] = combined_df['model'].map(display_names).fillna(combined_df['model'])

    print(f"  Combined result: {combined_df.shape}")
    dfs_all.append(combined_df)

print(f"\nBuilt {len(dfs_all)} combined DataFrames in dfs_all")

# Remove total_tokens and accuracy - but check indices first
print(f"dfs_all length before popping: {len(dfs_all)}")
if len(dfs_all) > 4:
    total_tokens = dfs_all.pop(4)
    print("Popped index 4 (total_tokens)")
else:
    print("WARNING: Cannot pop index 4, not enough DataFrames")

if len(dfs_all) > 6:
    dpsk_output = dfs_all.pop(6)
    print("Popped index 6 (dpsk_output)")
else:
    print("WARNING: Cannot pop index 6, not enough DataFrames")

print(f"dfs_all length after popping: {len(dfs_all)}")

## TMR Table (TODO Edit)

# 1) Build a dict of metric DataFrames
metric_dfs = { df['metric'].iloc[0] : df for df in dfs_all }

# 2) Extract median & p95
df_med = metric_dfs['timebetweentokens_median'].reset_index(drop=True)
df_p95 = metric_dfs['timebetweentokens_p95'].reset_index(drop=True)

# 3) Merge safely on keys
df_combined = pd.merge(
    df_med[['provider', 'model', 'input_size', 'value']],
    df_p95[['provider', 'model', 'input_size', 'value']],
    on=['provider', 'model'],
    suffixes=('_median', '_p95')
)

# 4) Per-request ratio
df_combined['p95_to_median'] = df_combined['value_p95'] / df_combined['value_median']

# 5) Group-level TMR: mean(p95) / mean(median)
summary_by_group = (
    df_combined
    .groupby(['provider', 'model'], as_index=False)
    .agg(
        p95_mean_tbt=('value_p95', 'mean'),
        median_mean_tbt=('value_median', 'mean')
    )
)
summary_by_group['tmr'] = summary_by_group['p95_mean_tbt'] / summary_by_group['median_mean_tbt']

# 6) Print or save
print("TMR Table (Grouped by provider, model)")
print(summary_by_group.to_markdown(index=False))

os.makedirs(graph_dir, exist_ok=True)
summary_by_group.to_csv(os.path.join(graph_dir, "tmr_df.csv"), index=False)

# Extract the raw TBT DataFrame
df_tbt = metric_dfs['timebetweentokens'].reset_index(drop=True)

# Initialize list to store parsed durations with providers
flattened = []

# Loop through each row, parse value list, associate with provider
for i, row in df_tbt.iterrows():
    val = row['value']
    provider = row['provider']

    if isinstance(val, str):
        try:
            parsed_list = ast.literal_eval(val)
            flattened.extend([(provider, float(x)) for x in parsed_list])
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing row {i}: {val[:50]}... Error: {e}")
            continue
    elif isinstance(val, list):
        flattened.extend([(provider, float(x)) for x in val])
    else:
        print(f"Skipping row {i}: unsupported type {type(val)}")

# Convert to DataFrame
df_flat = pd.DataFrame(flattened, columns=['provider', 'tbt'])

# Convert to milliseconds (if needed)
df_flat['tbt'] = df_flat['tbt'] * 1000

# Compute mean, std, CV per provider
cv_stats = (
    df_flat
    .groupby(['provider', 'model'])['tbt']
    .agg(['mean', 'std'])
    .rename(columns={'mean': 'mean_tbt', 'std': 'std_tbt'})
)
cv_stats['cv'] = cv_stats['std_tbt'] / cv_stats['mean_tbt']

# Join with your TMR summary
summary = summary_by_group.set_index('provider').join(cv_stats).reset_index()

# Show result
print("TBT TMR and CV")
print(summary.to_markdown(index=False))


# Optional: save
os.makedirs(graph_dir, exist_ok=True)
summary.to_csv(os.path.join(graph_dir, "tmr_tbt.csv"), index=False)

df = metric_dfs['timetofirsttoken'].reset_index(drop=True)

stats = (
    df.groupby(['provider','model'])['value']
    .quantile([0.5, 0.95, 0.99])
    .unstack(level=-1)
    .rename(columns={0.5:'median',0.95:'p95',0.99:'p99'})
)

stats['tmr'] = stats['p95'] / stats['median']

# Compute coefficient of variation
agg_stats = (
    df
    .groupby(['provider', 'model'])['value']
    .agg(['mean', 'std'])
)
agg_stats['cv'] = agg_stats['std'] / agg_stats['mean']

# Merge CV into stats
stats = stats.join(agg_stats)
print()
print("TTFT TMR and CV")
print(stats)