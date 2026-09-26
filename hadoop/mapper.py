import sys
import csv

for line in sys.stdin:
    line = line.strip()

    if not line:
        continue

    try:
        row = next(csv.reader([line]))

        # Skip header
        if row[0] == "Date":
            continue

        area = row[1]
        waste = float(row[3])

        print(f"{area}\t{waste}")

    except (ValueError, IndexError):
        continue