import torch
import pickle
import random

import data_utils as du
from model import Net

_, valid_data = du.load_labeled_data()
random.shuffle(valid_data)

# Load network, weights and word encodings
filename = input("File name: ")
with open("weights/" + filename + ".pkl", "rb") as f:
    word_to_idx = pickle.load(f)

net = Net(len(word_to_idx)).to(device=du.device, dtype=torch.float)
net.load_state_dict(torch.load("weights/" + filename + ".pt"))
net.eval()

for sentence, label in valid_data:
    # Encode sentence
    enc_val = du.encode_string(sentence, word_to_idx)
    enc_t = du.sentence_to_tensor(enc_val, 26)

    # Predict label
    _, pred_label = net(enc_t).cpu().detach().max(1)
    print(sentence)
    print("Predicted label: " + du.num_to_class[pred_label.item()])
    print("Actual label: " + du.num_to_class[label])
    input("Enter to continue...\n")
