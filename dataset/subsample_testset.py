import json
import random

with open('/share/hariharan/cloud_removal/metadata/v4/s2p_tx3_test_3k_na_split.json', 'r') as file:
  s2p_tx3_test_3k_na_split = json.load(file)

with open('/share/hariharan/cloud_removal/metadata/v4/s2p_tx3_na_3k_na_split.json', 'r') as file:
  s2p_tx3_na_3k_na_split = json.load(file)

# Group keys by ROI
test_roi_groups = {}
for key in s2p_tx3_test_3k_na_split.keys():
  roi = key.split('_')[0]
  if roi not in test_roi_groups:
    test_roi_groups[roi] = []
  test_roi_groups[roi].append(key)

na_roi_groups = {}
for key in s2p_tx3_na_3k_na_split.keys():
  roi = key.split('_')[0]
  if roi not in na_roi_groups:
    na_roi_groups[roi] = []
  na_roi_groups[roi].append(key)

# Set a fixed random seed
random.seed(42)

# Sample one key per ROI
test_sampled_keys = {roi: random.choice(keys) for roi, keys in test_roi_groups.items()}
na_sampled_keys = {roi: random.choice(keys) for roi, keys in na_roi_groups.items()}


# Create a subset of the dataset
test_subset = {key: s2p_tx3_test_3k_na_split[key] for key in test_sampled_keys.values()}
na_subset = {key: s2p_tx3_na_3k_na_split[key] for key in na_sampled_keys.values()}

# Save the subset to a new file
test_subset_file_path = '/share/hariharan/cloud_removal/metadata/v4/s2p_tx3_test_3k_na_split_1proi.json'
na_subset_file_path = '/share/hariharan/cloud_removal/metadata/v4/s2p_tx3_na_3k_na_split_1proi.json'
with open(test_subset_file_path, 'w') as subset_file:
  json.dump(test_subset, subset_file, indent=4)
with open(na_subset_file_path, 'w') as subset_file:
  json.dump(na_subset, subset_file, indent=4)