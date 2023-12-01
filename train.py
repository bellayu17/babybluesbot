import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import LLM

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import json file
with open('LLM.json', 'r') as f:
    intents = json.load(f)

# Set up lists for all_words, tags, and xy
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Define words to analyze and words to leave
ignore = ['?', ',', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore]
all_words = sorted(set(all_words)) # remove repetitive words
tags = sorted(set(tags)) # Leave only unique labels

# Create X train and y train dataset
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Preprocess dataset
class ChatBot(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.n_samples
    
# Define hyperparameters for model
BATCH_SIZE = 1
INPUT_SIZE = len(X_train[0])
HIDDEN_UNITS = 10
OUTPUT_SIZE = len(tags)
DATA = ChatBot()

train_loader = DataLoader(dataset=DATA, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,
                          num_workers=0)

# Import model.py and import customized hyperparameters
model = LLM(INPUT_SIZE,
            HIDDEN_UNITS,
            OUTPUT_SIZE).to(device)

# Create loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters(),
                             lr = 0.001)

# Train data
epochs = 1000
for epoch in range(epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        labels_pred = model(words)

        loss = loss_fn(labels_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f"epoch:{epoch} | loss: {loss.item():.5f}")

# Save model results
results = {"model_state": model.state_dict(),
           "input_size": INPUT_SIZE,
           "hidden_units": HIDDEN_UNITS,
           "output_size": OUTPUT_SIZE,
           "all_words": all_words,
           "tags": tags}

results_PATH = "results.pth"
torch.save(results, results_PATH)
print(f"Training completed")