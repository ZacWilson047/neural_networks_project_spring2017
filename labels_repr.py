from collections import defaultdict
import glob
import json
import os
import sys

LABELS_DIR = "/Users/zacwilson/data/iam_handwriting/rnnlib-iam-master/ascii"
SERIALIZE_FILENAME = "/Users/zacwilson/data/iam_handwriting/characters.json"
label_files = glob.glob(LABELS_DIR + "/*/*/*.txt")
print len(label_files)

characters = defaultdict(int)

for label_file in label_files:
    with open(label_file, "r") as filein:
        line = filein.readline()
        while True:
            if not line:
                raise ValueError("error 1: " + label_file)
            if line.strip() == "CSR:":
                break
            line = filein.readline()
        line = filein.readline().strip()
        if line != "":
            raise ValueError("error 2: " + label_file)
        line = filein.readline().strip()
        while line:
            line = line.strip()
            for char in line:
                characters[char] += 1
            line = filein.readline()

sorted_characters = sorted(characters.keys())
with open(SERIALIZE_FILENAME, "w") as fileout:
    json.dump(sorted_characters, fileout)

print len(characters.keys())
print characters
