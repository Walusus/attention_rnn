import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as mtr
import seaborn as sns
import math
import pickle

import data_utils as du
from model import Net

# TODO Unify comments
# Load and prepare dataset
train_data, valid_data = du.load_labeled_data()
word_to_idx, max_len = du.get_encodings(train_data + valid_data)

train_set = du.encode_dataset(train_data, word_to_idx)
valid_set = du.encode_dataset(valid_data, word_to_idx)

train_x, train_y = du.dataset_to_tensor(train_set, max_len)
valid_x, valid_y = du.dataset_to_tensor(valid_set, max_len)

# Create dataloaders
batch_size = 50
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

# Load network model
net = Net(100, len(word_to_idx), max_len).to(device=du.device, dtype=torch.float)

# Choose criterion and loss function
learning_rate = 1e-2
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

# TODO cleanup training function
# TODO tweak hyper parameters
# TODO remove test loader
# Train model
test_loss_track = []
train_loss_track = []
epochs_num = 20
train_batch_num = math.ceil(len(train_set) / batch_size)
for epoch in range(epochs_num):
    net.train()
    for batch_num, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss_track.append(loss.item())

        # Printing log each 5th batch
        if batch_num % 5 == 0:
            print(f"Epoch: {epoch + 1:d}/{epochs_num:d} ({(epoch + 1) / epochs_num * 100:.1f}%),"
                  f"\tMini-batch: {batch_num + 1:d} ({(batch_num + 1) / train_batch_num * 100:.1f}%),"
                  f"\tLoss: {loss.item():f}")

    # Test network every epoch
    net.eval()
    loss_sum = 0
    accuracy_sum = 0
    num = 0
    for batch_num, (inputs, labels) in enumerate(valid_loader):
        outputs = net(inputs)
        _, pred_labels = outputs.cpu().detach().max(1)
        loss_sum += loss_fn(outputs, labels).item()
        accuracy_sum += mtr.accuracy_score(labels.cpu(), pred_labels)
        num += 1

    accuracy = accuracy_sum / num
    loss = loss_sum / num
    test_loss_track.append(loss)
    print(f"Validation accuracy: {100 * accuracy:.1f}%,\tTest loss: {loss:f}")

# TODO fix plotting
# Plot learning results.
plt.figure(figsize=(8, 4))
plt.subplot(111)
plt.plot(np.linspace(1, epochs_num, num=len(train_loss_track)), train_loss_track, label="Train loss")
plt.plot(np.linspace(1, epochs_num, num=len(test_loss_track)), test_loss_track, label="Test loss")
plt.title(f"Epochs: {epochs_num:d}, batch size: {batch_size:d}, lr: {learning_rate:.1e}")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.savefig("plots/train_report.png")
plt.show()

# Test network
accuracy_sum = 0
num = 0
conf_mat = np.zeros((7, 7), dtype=np.int)
for batch_num, (inputs, labels) in enumerate(valid_loader):
    outputs = net(inputs)
    _, pred_labels = outputs.cpu().detach().max(1)
    accuracy_sum += mtr.accuracy_score(labels.cpu(), pred_labels)
    num += 1
    for i in range(len(pred_labels)):
        conf_mat[labels[i], pred_labels[i]] += 1

accuracy = accuracy_sum / num
print(f"\nValidation set accuracy: {100 * accuracy:.1f}%")

# Plot confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat, cmap="YlGnBu", annot=True, fmt="d", cbar=False, square=True)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.title(f"Validation set accuracy: {100 * accuracy:.1f}%")
plt.savefig("plots/conf_mat.png")
plt.show()

# Save model weights and word encodings
ans = input("Save model? [y/n]")
if ans is 'y':
    filename = input("Save as: ")
    torch.save(net.state_dict(), "weights/" + filename + ".pt")
    with open("weights/" + filename + ".pkl", "wb") as f:
        pickle.dump(word_to_idx, f)
