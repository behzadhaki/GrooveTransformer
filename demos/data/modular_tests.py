from data.src.dataLoaders import load_gmd_hvo_sequences
import os
from copy import deepcopy

os.chdir("../../")

dataset = load_gmd_hvo_sequences(dataset_setting_json_path="data/dataset_json_settings/4_4_Beats_gmd.json",
                                        subset_tag="test",
                                        force_regenerate=False)


hvo_seq = dataset[0]


hvo_seq_flattened = deepcopy(hvo_seq)
flat_seq = hvo_seq.flatten_voices()
hvo_seq_flattened.hvo = flat_seq

print(hvo_seq.hvo.shape)
print(hvo_seq.hvo[:5,:9])
print(hvo_seq_flattened.hvo.shape)
print(hvo_seq_flattened.hvo[:5,:9])

# hvo = hvo_seq.hvo


#cut out the offsets
# hvo_reduced = hvo[:,:18]
#
# hvo_flattened = hvo_seq.flatten_voices()
#
#
# hvo_seq_flattened = deepcopy(hvo_seq)
# hvo_seq_flattened.hvo = hvo_flattened







