#!/usr/bin/python3
import sys

from charset_normalizer import from_path
from utils import compute_md5

# Vérification des arguments
if len(sys.argv) != 4:
    print("Usage: python script.py <file_path> <start> <end>")
    sys.exit(1)

file_path = sys.argv[1]
start     = int(sys.argv[2])
end       = int(sys.argv[3])

match = from_path(file_path).best()
if match is None:
    raise ValueError("Could not detect encoding of the file.")
text = str(match)


extracted_text = text[start:end]
computed_md5 = compute_md5(extracted_text)
print(f"{computed_md5}: '{extracted_text}'")
