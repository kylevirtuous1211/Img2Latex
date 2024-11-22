import os
import pandas as pd
import chardet
import pickle as pkl
from collections import Counter
from os.path import join
import argparse

# Define special tokens
START_TOKEN = "<s>"
END_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

class Vocab(object):
    def __init__(self):
        self.sign2id = {
            START_TOKEN: 0,
            END_TOKEN: 1,
            PAD_TOKEN: 2,
            UNK_TOKEN: 3
        }
        self.id2sign = {idx: token for token, idx in self.sign2id.items()}
        self.length = 4  # Starting index after special tokens

    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1

    def __len__(self):
        return self.length

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def tokenize_formula(formula):
    """
    Tokenize a LaTeX formula.
    This tokenizer splits LaTeX commands, symbols, and numbers appropriately.
    """
    import re
    # Regex pattern to match LaTeX commands, braces, and other symbols
    pattern = r'\\[a-zA-Z]+|\{|\}|\^|\_|\$|[a-zA-Z0-9]+|[^a-zA-Z0-9\s]'
    tokens = re.findall(pattern, formula)
    return tokens

def build_vocab(data_dir, min_count=10):
    """
    Traverse training formulas to build vocab
    and store the vocab in a pickle file.
    """
    vocab = Vocab()
    counter = Counter()

    # Path to im2latex_train.csv
    csv_path = join(data_dir, 'im2latex_train.csv')

    # Check if CSV file exists
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return None, None

    # Detect encoding
    encoding = detect_file_encoding(csv_path)
    print(f"Detected encoding for CSV file: {encoding}")

    # Read formulas from CSV file
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None

    if 'formula' not in df.columns:
        print("Error: 'formula' column not found in CSV.")
        return None, None

    # Drop rows with missing formulas
    formulas = df['formula'].dropna().tolist()
    print(f"Total formulas loaded: {len(formulas)}")

    # Tokenize formulas and update counter
    for formula in formulas:
        tokens = tokenize_formula(formula)
        counter.update(tokens)

    print(f"Total unique tokens before filtering: {len(counter)}")

    # Add tokens with frequency >= min_count to vocab
    for token, count in counter.most_common():
        if count >= min_count:
            vocab.add_sign(token)

    print(f"Vocabulary size after applying min_count={min_count}: {len(vocab)}")

    # Path to save vocab.pkl
    vocab_file = join(data_dir, 'vocab.pkl')
    print("Writing Vocab File to", vocab_file)
    with open(vocab_file, 'wb') as w:
        pkl.dump(vocab, w)

    return vocab, counter

def load_vocab(data_dir):
    """
    Load the vocabulary from vocab.pkl.
    """
    vocab_file = join(data_dir, 'vocab.pkl')
    if not os.path.isfile(vocab_file):
        print(f"Error: Vocab file not found at {vocab_file}")
        return None

    with open(vocab_file, 'rb') as f:
        vocab = pkl.load(f)
    print(f"Loaded vocab with {len(vocab)} tokens!")
    return vocab

def main():
    parser = argparse.ArgumentParser(description="Build Vocabulary for Im2Latex Dataset")
    parser.add_argument("--data_path", type=str, default=".\data", help="Directory containing im2latex_train.csv")
    parser.add_argument("--min_count", type=int, default=10, help="Minimum frequency for tokens to be included in vocab")
    args = parser.parse_args()

    vocab, counter = build_vocab(args.data_path, min_count=args.min_count)

    if vocab and counter:
        # Optionally, print some token counts
        print("\nSample Token Counts (Top 20):")
        for token, count in counter.most_common(20):
            if count >= args.min_count:
                print(f"{token}: {count}")

if __name__ == "__main__":
    main()
