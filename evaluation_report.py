import json
from collections import Counter
import os

# Iterate over all JSON files
for file in sorted(os.listdir("output")):
    if file.endswith(".json"):
        file = os.path.join("output", file)
        file = open(file, "r")

        # Load JSON data
        print(f"\nReading: {file.name}...")
        data = json.load(file)

        # Get count of scores, sorted by count
        scores = [score["score"] for score in data["scores"]]
        counter = Counter(scores)
        sorted_counter = sorted(counter.items())
        print(f"Scores: {sorted_counter}")

        # Calculate total count
        total_count = sum(counter.values())
        print(f"Total count: {total_count}")

        # Scores as list
        print(scores)

        # Print 1 score per line - easy to paste into Excel to analyze
        for score in scores:
            print(score)
