import random
import json
import torch
from model import LLM
from nltk_utils import bag_of_words, tokenize

# Set up the device to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Open up and import the json data
with open('LLM.json', 'r') as json_data:
    intents = json.load(json_data)

# Import training results
results_PATH = "results.pth"
results = torch.load(results_PATH)

INPUT_SIZE = results["input_size"]
OUTPUT_SIZE = results["output_size"]
HIDDEN_UNITS = results["hidden_units"]
all_words = results['all_words']
tags = results['tags']
model_state = results["model_state"]

# Load the model
model = LLM(INPUT_SIZE,
            HIDDEN_UNITS,
            OUTPUT_SIZE).to(device)
model.load_state_dict(model_state)
model.eval()

# Create the chatbot
bot = "BabyBlues"

def export_answer(msg):
  sentence = tokenize(msg)
  X = bag_of_words(sentence, all_words)
  X = X.reshape(1, X.shape[0])
  X = torch.from_numpy(X).to(device)

  y_preds = model(X)
  _, predicted = torch.max(y_preds, dim=1)

  tag = tags[predicted.item()]

  probs = torch.softmax(y_preds, dim=1)
  prob = probs[0][predicted.item()]
  if prob.item() > 0.75:
    for intent in intents['intents']:
      if tag == intent['tag']:
        return random.choice(intent['responses'])
  else:
    return "Could you elaborate more?"

if __name__ == "__main__":
    print("I'm BabyBlues, and I cared about your health and wellness!")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        response = export_answer(sentence)
        print(response)

