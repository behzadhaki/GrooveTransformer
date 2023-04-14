
from data import load_original_gmd_dataset_pickle
import os
from data.src.dataLoaders import MonotonicGrooveTokenizedDataset

os.chdir("../../")


"""
1. Load pre-token dataset (GMD at 96)
2. Tokenize
3. Convert to Dataloader
4. Print out some examples, try enumerating
"""
tokenized_dataset = MonotonicGrooveTokenizedDataset(
        dataset_setting_json_path="data/dataset_json_settings/BeatsAndFills_gmd_96.json",
        subset_tag="test")


from torch.utils.data import DataLoader
from data.src.dataLoaders import custom_collate_fn

data_loader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

for batch_idx, batch in enumerate(data_loader):
    print(f"\n\nBatch {batch_idx + 1}:")
    print(f"Batch shape: {len(batch)}")
    print(f"Batch content (first 3 elements):")
    print(type(batch))
    print(len(batch))
    print(type(batch[0]))
    print(batch[0].shape)
    #print(batch[0][0][:5][:9])

    # Stop after a few batches for testing purposes
    if batch_idx >= 2:
        break