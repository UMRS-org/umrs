#!/usr/bin/python3
import json
import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} file.json")
    sys.exit(1)

filename = sys.argv[1]

try:
    with open(filename) as f:
        data = json.load(f)
    print("Valid JSON!")
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Invalid JSON: {e}")

