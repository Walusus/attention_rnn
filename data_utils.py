import os
import json
import torch

# TODO add comments
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_to_num = {"AddToPlaylist": 0,
                "BookRestaurant": 1,
                "GetWeather": 2,
                "PlayMusic": 3,
                "RateBook": 4,
                "SearchCreativeWork": 5,
                "SearchScreeningEvent": 6}

num_to_class = {0: "AddToPlaylist",
                1: "BookRestaurant",
                2: "GetWeather",
                3: "PlayMusic",
                4: "RateBook",
                5: "SearchCreativeWork",
                6: "SearchScreeningEvent"}


def string_to_word_list(string):
    word_list = string.split()
    stripped_list = []

    for word in word_list:
        strp_word = word.strip(" .:?!,").rstrip("'s").lower()
        if strp_word.replace('/', '').replace(':', '').replace('-', '').isnumeric():
            strp_word = " num "
        stripped_list.append(strp_word)

    return stripped_list


def load_labeled_data():
    valid_s = []
    train_s = []

    root, _, files = next(os.walk("datasets/"))
    for filename in files:
        with open(root + filename, 'r') as f:
            _, data = json.load(f).popitem()

            label = class_to_num[filename.strip(".json").split("_")[1]]

            for s in data:
                sentence = ''.join(part['text'] for part in s['data'])

                if filename.startswith("validate"):
                    valid_s.append((sentence, label))
                else:
                    train_s.append((sentence, label))

    return train_s, valid_s


def get_encodings(dataset):
    vocab = set()
    max_len = 0

    for sentence, _ in dataset:
        word_list = string_to_word_list(sentence)
        max_len = len(word_list) if len(word_list) > max_len else max_len
        vocab.update(word_list)

    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx[" pad "] = 0
    return word_to_idx, max_len


def encode_string(sentence, encodings):
    return [encodings[word] for word in string_to_word_list(sentence)]


def encode_dataset(dataset, encodings):
    encoded_data = []
    for value, label in dataset:
        encoded_data.append((encode_string(value, encodings), label))

    return encoded_data


def sentence_to_tensor(sentence, max_len, label=None):
    # Pad sentence with zeros
    sentence += [0 for i in range(max_len - len(sentence))]
    ts = torch.tensor(sentence, dtype=torch.long, device=device).unsqueeze(0)

    if label is not None:
        tl = torch.tensor(label, dtype=torch.long, device=device).unsqueeze(0)
        return ts, tl
    else:
        return ts


def dataset_to_tensor(dataset, max_len):
    sentence_list = []
    label_list = []
    for value, label in dataset:
        v_t, l_t = sentence_to_tensor(value, max_len, label)
        sentence_list.append(v_t)
        label_list.append(l_t)

    return torch.cat(sentence_list), torch.cat(label_list)
