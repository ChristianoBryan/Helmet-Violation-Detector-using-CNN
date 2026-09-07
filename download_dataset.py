from pathlib import Path

# Training labels folder
labels_dir = Path("helmetviolation/train_balanced/labels")

# Correct class mapping from the official data.yaml
class_names = {
    0: "Plate",
    1: "WithHelmet",
    2: "WithoutHelmet"
}

# Counters
image_counts = {0: 0, 1: 0, 2: 0}
annotation_counts = {0: 0, 1: 0, 2: 0}

# Go through every label file
for label_file in labels_dir.glob("*.txt"):
    classes_in_image = set()

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if not parts:
                continue

            class_id = int(parts[0])

            if class_id in annotation_counts:
                annotation_counts[class_id] += 1
                classes_in_image.add(class_id)

    # Count the image only once per class
    for class_id in classes_in_image:
        image_counts[class_id] += 1


# Display results
print("\n=== TRAINING DATASET CLASS DISTRIBUTION ===\n")

for class_id, name in class_names.items():
    print(f"{name}:")
    print(f"  Images containing class : {image_counts[class_id]}")
    print(f"  Total annotations       : {annotation_counts[class_id]}")
    print()

print("=== IMBALANCE ===")

max_count = max(image_counts.values())

for class_id, name in class_names.items():
    ratio = max_count / image_counts[class_id] if image_counts[class_id] else 0
    print(f"{name}: {ratio:.2f}x compared to the smallest class")