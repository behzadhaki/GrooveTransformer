
from data import load_mega_dataset_hvo_sequences

# Load GMD Dataset in `HVO_Sequence` format using a single command
dataset = load_mega_dataset_hvo_sequences(
    dataset_setting_json_path="./data/dataset_json_settings/4_4_Beats_mega_beats.json",
    subset_tag="train")

# =================================================================================================
# Load dataset as torch.utils.data.Dataset
from data import MonotonicGrooveDataset

# load dataset as torch.utils.data.Dataset
training_dataset = MonotonicGrooveDataset(
    dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
    subset_tag="train",
    max_len=32,
    tapped_voice_idx=2,
    collapse_tapped_sequence=False,
    load_as_tensor=True,
    sort_by_metadata_key="loop_id",
    genre_loss_balancing_beta=0.5,
    voice_loss_balancing_beta=0.5
)

training_dataset.__getitem__(0)
training_dataset.get_voice_counts()
training_dataset.get_genre_distributions_dict()
# training_dataset.visualize_global_hit_count_ratios_heatmap()
#training_dataset.visualize_genre_distributions(show_inverted_weights=True)
# =================================================================================================

# use the above dataset in the training pipeline, you need to use torch.utils.data.DataLoader
from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)

epochs = 10
for epoch in range(epochs):
    # in each epoch we iterate over the entire dataset
    for batch_count, (inputs, outputs, indices) in enumerate(train_dataloader):
        print(f"Epoch {epoch} - Batch #{batch_count} - inputs.shape {inputs.shape} - "
              f"outputs.shape {outputs.shape} - indices.shape {indices.shape} ")