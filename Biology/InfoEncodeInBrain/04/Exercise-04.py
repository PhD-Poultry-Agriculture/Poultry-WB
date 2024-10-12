# Author: German Shiklov
# ID: 317634517
# Encoding in Brain - Exercise 4.
#%%
from collections import defaultdict
import string
import math
import os
import gzip
import zipfile

file_path = '/Users/gshiklov/Documents/Projects/GitProjects/Jeremaiha/GG-Machine/InfoEncodeInBrain/04/MOBY-DICK.txt'
#%% Part 1: Entropy of symbols
def simulate_part1(file_path):
    symbol_counts = defaultdict(int)
    
    with open(file_path, 'r') as file:
        text = file.read()
        for char in text:
            if char.isalpha():
                symbol_counts['alphabet'] += 1
            elif char.isdigit():
                symbol_counts['numbers'] += 1
            elif char in string.punctuation:
                symbol_counts['punctuation'] += 1
            elif char.isspace():
                symbol_counts['spaces'] += 1
            else:
                symbol_counts['other'] += 1
    
    total_chars = sum(symbol_counts.values())
    print("Total number of characters:", total_chars)
    print("\nCharacter\tCount\tPercentage")
    print("---------------------------------")
    for char, count in sorted(symbol_counts.items()):
        percentage = (count / total_chars) * 100
        print(f"{char}\t\t{count}\t{percentage:.2f}%")

    total_symbols = sum(symbol_counts.values())
    entropy = 0
    probabilities = {symbol: count / total_symbols for symbol, count in symbol_counts.items()}
    entropy = -sum(prob * math.log2(prob) for prob in probabilities.values())
    print("Estimated entropy per character (in bits):", entropy)

simulate_part1(file_path)

# %% Part 2 - Entropy per pair: (’aa’, ’ab’, ’ac’,.., ,’a:’, .. , ’ba’, ....).
def simulate_part2(file_path):
    symbol_counts = defaultdict(int)
    total_pairs = 0
    
    with open(file_path, 'r') as file:
        text = file.read().lower()
        for i in range(len(text) - 1):
            pair = text[i:i+2]
            symbol_counts[pair] += 1
            total_pairs += 1
    
    print("Total number of pairs:", total_pairs)
    print("\nPair\t\tCount\tPercentage")
    print("---------------------------------")
    for pair, count in sorted(symbol_counts.items()):
        percentage = (count / total_pairs) * 100
        print(f"{pair}\t\t{count}\t{percentage:.2f}%")
    entropy = 0
    probabilities = {pair: count / total_pairs for pair, count in symbol_counts.items()}
    entropy = -sum(prob * math.log2(prob) for prob in probabilities.values())
    print("Estimated entropy per character (in bits):", entropy)

simulate_part2(file_path)
# %% Part 3 - Conditional Entropy
def simulate_part3(file_path):
    symbol_counts = defaultdict(lambda: defaultdict(int))
    total_pairs = 0
    
    with open(file_path, 'r') as file:
        text = file.read().lower()
        for i in range(len(text) - 1):
            current_symbol = text[i]
            next_symbol = text[i+1]
            symbol_counts[current_symbol][next_symbol] += 1
            total_pairs += 1
    
    entropy_sum = 0
    total_symbols = len(symbol_counts.keys())
    
    for symbol, next_symbols in symbol_counts.items():
        symbol_total_count = sum(next_symbols.values())
        symbol_entropy = 0
        
        for next_symbol, count in next_symbols.items():
            probability = count / symbol_total_count
            symbol_entropy += probability * math.log2(1 / probability)
        
        entropy_sum += symbol_entropy * (symbol_total_count / total_pairs)
    
    print("Estimated average entropy per symbol conditioned on the previous symbol (in bits):", entropy_sum)

simulate_part3(file_path)
# %% Part 4 -  Entropy

def simulation_part4(file_path):
    sequence_counts = defaultdict(int)
    total_sequences = 0
    
    with open(file_path, 'r') as file:
        text = file.read().lower()
        
        for i in range(len(text) - 9):
            sequence = text[i:i+10]
            if not sequence.isalpha():
                continue
            sequence_counts[sequence] += 1
            total_sequences += 1
    entropy = 0
    
    for sequence, count in sequence_counts.items():
        probability = count / total_sequences
        entropy -= probability * math.log2(probability)
    
    print(f"Estimated entropy for 10-letter sequences (in bits): {entropy}")

simulation_part4(file_path)
# %% Part 5 - Compression via gzip
def simulation_part5(file_path):
    compressed_file_path = file_path + '.gz'
    with open(file_path, 'rb') as original_file:
        with gzip.open(compressed_file_path, 'wb') as compressed_file:
            compressed_file.writelines(original_file)
        
    original_size = os.path.getsize(file_path)
    compressed_size = os.path.getsize(compressed_file_path)
        
    with open(file_path, 'r', encoding='utf-8') as f:
        text_length = len(f.read())
    
    bits_per_character = (compressed_size * 8) / text_length
    print(f"Bits per character required after compression: {bits_per_character}")
    os.remove(compressed_file_path)

simulation_part5(file_path)
# %% Part 5 - Compression via zip
def simulation_part5_zip(file_path):
    compressed_file_path = file_path + '.zip'
    with zipfile.ZipFile(compressed_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))
    
    original_size = os.path.getsize(file_path)
    compressed_size = os.path.getsize(compressed_file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text_length = len(f.read())
        
    bits_per_character = (compressed_size * 8) / text_length    
    print(f"Bits per character required after ZIP compression: {bits_per_character}")
    os.remove(compressed_file_path)

simulation_part5_zip(file_path)