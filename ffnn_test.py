import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax(dim = 0) # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        # setting dim = 0 since output is a 1-D score vector
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        # hidden h \in R^{|h|}
        hidden_invector = self.W1(input_vector)
        hidden = self.activation(hidden_invector)

        # [to fill] obtain output layer representation
        # logits z output \in R^{|Y|}
        output_layer = self.W2(hidden)

        # [to fill] obtain probability dist.
        # log-probs y \in R^{|Y|}, sum(exp(.)) = 1
        log_predicted_vector = self.softmax(output_layer)

        return log_predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    

    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    # optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Saving best models for testing
    os.makedirs("results", exist_ok=True)
    best_val = -1.0
    best_epoch = -1
    best_path = f"results/ffnn_best.pt"


    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        # optimizer.zero_grad() # included for training, inside minibatch loop for testing
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        print("Training loss: {}".format(loss))

        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        
        model.eval()                      
        with torch.no_grad():             
            minibatch_size = 16 
            N = len(valid_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                # optimizer.zero_grad() # included while training
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size

        val_acc = correct / total
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, val_acc))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        print("Validation loss: {}".format(loss))

        print(f"Best validation: epoch {best_epoch}  acc={best_val:.4f}")

        # save best
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            torch.save({
                "epoch": best_epoch,
                "val_acc": best_val,
                "model": model.state_dict(),
                "opt": optimizer.state_dict()
            }, best_path)
            print(f"[saved best] {best_path} (epoch={best_epoch}, val_acc={best_val:.4f})")

        model.train()                     # <--- return to train mode for next epoch
            
    print(f"Training finished. Best validation: epoch {best_epoch}  acc={best_val:.4f}")


    # ===== TEST INFERENCE (runs only if --test_data provided) =====
    if args.test_data != "to fill":
        print(f"========== Testing on {args.test_data} ==========")

        # Load best checkpoint (if present)
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            print(f"Loaded best checkpoint from: {best_path}")
        else:
            print("Warning: best checkpoint not found; using current in-memory weights.")

        # Load raw test JSON
        with open(args.test_data, "r", encoding="utf-8") as f:
            test_raw = json.load(f)

        # Vectorize with training vocab (word2index)
        def bow_vectorize(toks, word2index, unk_token=unk):
            v = torch.zeros(len(word2index))
            for w in toks:
                idx = word2index.get(w, word2index[unk_token])
                v[idx] += 1
            return v

        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for item in test_raw:
                toks = item.get("text", "").split()
                x = bow_vectorize(toks, word2index)
                logp = model(x)                 # (5,) log-probs
                yhat = int(torch.argmax(logp))  # 0..4
                preds.append(yhat + 1)          # convert to stars 1..5
                gold.append(int(item["stars"])) # test.json has labels

        # Save predictions
        os.makedirs("results", exist_ok=True)
        out_csv = "results/test_predictions.csv"
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("prediction\n")
            for p in preds:
                f.write(f"{p}\n")
        print("Saved test predictions to:", out_csv)

        # Test accuracy
        correct = sum(int(g == p) for g, p in zip(gold, preds))
        test_acc = correct / len(preds) if preds else 0.0
        print(f"Test accuracy: {test_acc:.4f}  ({correct}/{len(preds)})")
        with open("results/test_metrics.txt", "w") as f:
            f.write(f"test_accuracy,{test_acc:.6f}\ncorrect,{correct}\ntotal,{len(preds)}\n")
