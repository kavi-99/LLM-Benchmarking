# TMR, Coefficient of Variation

import pandas as pd
import glob
import os
import ast

exp_dirs = {
    "aws": "experiments/exp_20250620_110503_06a15fc4",
    "aws_5k": "experiments/aws_1k",
    "aws_10k": "experiments/aws_5k",
    "vllm_azure": "experiments/exp_20250620_013929_b5497e18",
    "azure_5k": "experiments/azure_5k_new",
    "azure_10k": "experiments/azure_10k_new",
    "togetherai": "experiments/exp_20250626_050759_3bb60a5c",
    "togetherai_5k": "experiments/togetherai_5k",
    "togetherai_10k": "experiments/togetherai_10k"
}

graph_dir = "experiments/zexp_5.2_output"

# 2) Read each folder’s CSVs into a dict of lists-of-DataFrames
dfs = {}
all_dfs = {}
first_provider = list(exp_dirs.keys())[0]   # e.g. "aws"
last_provider = list(exp_dirs.keys())[-1]

skip_idxs = {0, 3, 7}
for name, folder in exp_dirs.items():
    pattern = os.path.join(folder, "*.csv")
    paths   = glob.glob(pattern)
    print(name)
    for p in paths:
        try:
            print(p)
            df = pd.read_csv(p)
        except Exception as e:
            print(f"Error in file: {p}")
            raise

    all_dfs[name] = [pd.read_csv(p) for p in paths]
    # dfs[name] = [
    #     df
    #     for idx, df in enumerate(all_dfs[name])
    #     if idx not in skip_idxs
    # ]
    dfs[name] = []
    for idx, df in enumerate(all_dfs[name]):
        if idx in skip_idxs:
            continue

        # if this is the first provider, take only the first 100 rows
        if name == first_provider:
            dfs[name].append(df.head(100))
        # if name == last_provider:
        #     dfs[name].append(df.head(201))
        elif name == "togetherai":
            dfs[name].append(df.head(200))
        elif name == "vllm_azure":
            dfs[name].append(df.head(583))
        else:
            dfs[name].append(df)
    for i, df in enumerate(dfs[name]):
        print(i, df['metric'].unique())
    print(f"{name}: loaded {len(dfs[name])} CSV(s) from {folder}")

# 3) Build dfs_all by zipping in the same provider-order
provider_order = list(exp_dirs.keys())  # e.g. ["google", "openai", …, "vllm"]
# get list-of-lists in order:
list_of_df_lists = [dfs[p] for p in provider_order]

# 4) Concatenate the i-th file of each provider across all providers
dfs_all = [
    pd.concat(dfs_tuple, ignore_index=True)
    for dfs_tuple in zip(*list_of_df_lists)
]

print(f"Built {len(dfs_all)} combined DataFrames in dfs_all")

# Remove total_tokens and accuracy as before
total_tokens = dfs_all.pop(4)
dpsk_output = dfs_all.pop(6)
# for i, df in enumerate(dfs_all):
#     print(df.head(1))
# #     # want to remove entries with model = "common-model-small"
#     print(df['metric'].unique())
#     print(df['provider'].unique())
#     print(df['model'].unique())
#     print(df['max_output'].unique())
#     print(df['input_size'].unique())
#     print(f"combined #{i}: {df.shape}")
#     print("-----------------")

for i, df in enumerate(dfs_all):
    print(df.head(1))
# want to remove entries with model = "common-model-small"
    print(df['metric'].unique())
    print(df['provider'].unique())
    print(df['model'].unique())
    print(df['max_output'].unique())
    print(f"combined #{i}: {df.shape}")
    print("-----------------")

# 1) Build a dict of your metric‐DataFrames by name

# 1) Build a dict of metric DataFrames
metric_dfs = { df['metric'].iloc[0] : df for df in dfs_all }

# 2) Extract median & p95
df_med = metric_dfs['timebetweentokens_median'].reset_index(drop=True)
df_p95 = metric_dfs['timebetweentokens_p95'].reset_index(drop=True)

# 3) Merge safely on keys
df_combined = pd.merge(
    df_med[['provider', 'model', 'max_output', 'value']],
    df_p95[['provider', 'model', 'max_output', 'value']],
    on=['provider', 'model', 'max_output'],
    suffixes=('_median', '_p95')
)
print(df_combined[['value_p95', 'value_median']].dtypes)
df_combined['value_p95'] = pd.to_numeric(df_combined['value_p95'], errors='coerce')
print(df_combined[['value_p95', 'value_median']].dtypes)
# 4) Per-request ratio
df_combined['p95_to_median'] = df_combined['value_p95'] / df_combined['value_median']

# 5) Group-level TMR: mean(p95) / mean(median)
summary_by_group = (
    df_combined
    .groupby(['provider', 'max_output'], as_index=False)
    .agg(
        p95_mean_tbt=('value_p95', 'mean'),
        median_mean_tbt=('value_median', 'mean')
    )
)
summary_by_group['tmr'] = summary_by_group['p95_mean_tbt'] / summary_by_group['median_mean_tbt']

# 6) Print or save
print("TMR Table (Grouped by provider, max_output)")
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
    .groupby(['provider', 'max_output'])['tbt']
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
    df.groupby(['provider','max_output'])['value']
    .quantile([0.5, 0.95, 0.99])
    .unstack(level=-1)
    .rename(columns={0.5:'median',0.95:'p95',0.99:'p99'})
)

stats['tmr'] = stats['p95'] / stats['median']

# Compute coefficient of variation
agg_stats = (
    df
    .groupby(['provider', 'max_output'])['value']
    .agg(['mean', 'std'])
)
agg_stats['cv'] = agg_stats['std'] / agg_stats['mean']

# Merge CV into stats
stats = stats.join(agg_stats)
print()
print("TTFT TMR and CV")
print(stats)