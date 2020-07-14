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


def string_to_word_list(string):
    word_list = string.split()
    stripped_list = []

    for word in word_list:
        strp_word = word.strip(" .:?!,").rstrip("'s").lower()
        if strp_word.replace('/', '').replace(':', '').replace('-', '').isnumeric():
            strp_word = "num"
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

    for sentence, _ in dataset:
        word_list = string_to_word_list(sentence)
        vocab.update(word_list)

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    return word_to_idx


def encode_string(sentence, encodings):
    return [encodings[word] for word in string_to_word_list(sentence)]


def encode_dataset(dataset, encodings):
    encoded_data = []
    for value, label in dataset:
        encoded_data.append((encode_string(value, encodings), label))

    return encoded_data


def sentence_to_tensors(sentence, label=None):
    ts = torch.tensor(sentence, dtype=torch.float32, device=device).unsqueeze(0)

    if label is not None:
        tl = torch.tensor(label, dtype=torch.float32, device=device).unsqueeze(0)
        return ts, tl
    else:
        return ts
