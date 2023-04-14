import numpy as np
import os

hvo_size = 3
data = [("hit", np.random.rand(hvo_size)),
       ("hit", np.random.rand(hvo_size)),
       ("delta", np.random.rand(hvo_size)),
       ("measure", np.random.rand(hvo_size))]



def create_vocab_dictionary(data):
    token_types = [token_type for token_type, _ in data]
    unique_token_types = sorted(set(token_types))
    vocab = {token_type: i + 1 for i, token_type in enumerate(unique_token_types)}
    return vocab

def encode_tokenized_data(data, vocab):
    encoded_data = [np.concatenate((np.array([vocab[token_type]]), hvo_seq), axis=0) for token_type, hvo_seq in data]
    return encoded_data

if __name__ == "__main__":
    # for i in data:
    #     print(i)
    #
    # vocab = create_vocab_dictionary(data)
    # print("\nDictionary:")
    # print(vocab)
    #
    # encoded_data = encode_tokenized_data(data, vocab)
    # print("\nEncoded Data:")
    # for i in encoded_data:
    #     print(i)
    # print(encoded_data[0][1])



    print("Starting..")
    os.chdir("../../")

    from data import MonotonicGrooveTokenizedDataset


    test_dataset = MonotonicGrooveTokenizedDataset(
        dataset_setting_json_path="data/dataset_json_settings/BeatsAndFills_gmd_96.json",
        subset_tag="test")

    vocab = test_dataset.get_vocab_dictionary()
    print(vocab)

    test_output = test_dataset.outputs[3]

    print(len(test_dataset.hvo_sequences))
    print(len(test_dataset.outputs))
    print(len(test_dataset.inputs))