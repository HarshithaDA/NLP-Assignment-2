import json
import torch
import pickle
import string
from tqdm import tqdm

# Load test data
def load_test_data(test_file):
    with open(test_file) as f:
        data = json.load(f)
    test_data = []
    for elt in data:
        test_data.append((elt["text"].split(), int(elt["stars"] - 1)))
    return test_data

# Test function
def test_model(model, test_data, word_embedding):
    model.eval()
    correct = 0
    total = 0
    predictions = []

    for input_words, gold_label in tqdm(test_data):
        input_words = " ".join(input_words)
        input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
        output = model(vectors)

        predicted_label = torch.argmax(output)
        predictions.append(predicted_label.item())
        correct += int(predicted_label == gold_label)
        total += 1

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return predictions, accuracy

if __name__ == "__main__":
    import argparse
    from rnn_siddhi import RNN  # Make sure the training script is importable

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True, help="Path to test data JSON")
    parser.add_argument("--model_path", required=True, help="Path to saved model .pt file")
    parser.add_argument("--word_embedding", default="./word_embedding.pkl", help="Path to word embedding pickle file")
    parser.add_argument("--hidden_dim", type=int, required=True, help="Hidden dimension of RNN")
    args = parser.parse_args()

    # Load word embeddings
    word_embedding = pickle.load(open(args.word_embedding, "rb"))

    # Load trained model
    model = RNN(50, args.hidden_dim)  # Ensure input_dim=50 as in training
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Load test data
    test_data = load_test_data(args.test_data)

    # Run test
    predictions, accuracy = test_model(model, test_data, word_embedding)

    # Optionally, save predictions to a file
    with open("test_predictions.json", "w") as f:
        json.dump(predictions, f)
