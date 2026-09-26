import sys

current_area = None
total_waste = 0.0

for line in sys.stdin:
    line = line.strip()

    if not line:
        continue

    try:
        area, waste = line.split("\t", 1)
        waste = float(waste)

        if current_area == area:
            total_waste += waste

        else:
            if current_area is not None:
                print(f"{current_area}\t{total_waste:.2f}")

            current_area = area
            total_waste = waste

    except ValueError:
        continue

if current_area is not None:
    print(f"{current_area}\t{total_waste:.2f}")