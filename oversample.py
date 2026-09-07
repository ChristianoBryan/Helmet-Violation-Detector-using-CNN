from pathlib import Path
import cv2
import albumentations as A
import random
import shutil

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path("helmetviolation")

SOURCE_IMAGES = BASE_DIR / "train" / "images"
SOURCE_LABELS = BASE_DIR / "train" / "labels"

OUTPUT_IMAGES = BASE_DIR / "train_balanced" / "images"
OUTPUT_LABELS = BASE_DIR / "train_balanced" / "labels"

TARGET_WITH_HELMET = 500
TARGET_WITHOUT_HELMET = 500

# Official class IDs
PLATE = 0
WITH_HELMET = 1
WITHOUT_HELMET = 2


# ============================================================
# AUGMENTATION
# ============================================================

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),

        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.05, 0.05),
            rotate=(-5, 5),
            p=0.7
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        A.GaussNoise(p=0.2),

        A.Blur(
            blur_limit=3,
            p=0.15
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3
    )
)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)


# ============================================================
# COPY ORIGINAL TRAINING DATA
# ============================================================

print("Copying original training data...")

for image_path in SOURCE_IMAGES.iterdir():

    if not image_path.is_file():
        continue

    label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    shutil.copy2(
        image_path,
        OUTPUT_IMAGES / image_path.name
    )

    shutil.copy2(
        label_path,
        OUTPUT_LABELS / label_path.name
    )


# ============================================================
# READ ALL TRAINING IMAGES
# ============================================================

with_helmet_images = []
without_helmet_images = []

for label_path in SOURCE_LABELS.glob("*.txt"):

    with open(label_path, "r") as f:
        lines = f.readlines()

    classes = set()

    for line in lines:

        parts = line.strip().split()

        if len(parts) >= 5:
            classes.add(int(parts[0]))

    # Find image
    image_path = None

    for extension in [
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG"
    ]:

        candidate = SOURCE_IMAGES / (
            label_path.stem + extension
        )

        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        continue

    # Add image to WithHelmet pool
    if WITH_HELMET in classes:
        with_helmet_images.append(
            (image_path, label_path)
        )

    # Add image to WithoutHelmet pool
    if WITHOUT_HELMET in classes:
        without_helmet_images.append(
            (image_path, label_path)
        )


print()
print("Original helmet image counts:")
print(f"WithHelmet    : {len(with_helmet_images)}")
print(f"WithoutHelmet : {len(without_helmet_images)}")
print()


# ============================================================
# AUGMENT FUNCTION
# ============================================================

def augment_class(
    image_list,
    target_count,
    class_name
):

    original_count = len(image_list)

    if original_count == 0:
        print(
            f"ERROR: No images found for {class_name}."
        )
        return

    if original_count >= target_count:

        print(
            f"{class_name}: already has "
            f"{original_count} images."
        )

        return

    needed = target_count - original_count

    print(
        f"Generating {needed} additional "
        f"{class_name} images..."
    )

    generated = 0

    while generated < needed:

        image_path, label_path = random.choice(
            image_list
        )

        # Read image
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        # Read YOLO labels
        bboxes = []
        class_labels = []

        with open(label_path, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                cls = int(parts[0])

                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                bboxes.append(
                    [x, y, w, h]
                )

                class_labels.append(cls)

        if not bboxes:
            continue

        # Augment image + ALL its labels
        try:

            result = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )

        except Exception:
            continue

        augmented_image = result["image"]
        augmented_boxes = result["bboxes"]
        augmented_classes = result["class_labels"]

        if len(augmented_boxes) == 0:
            continue

        # Make sure the target helmet class
        # survived the augmentation.
        if class_name == "WithHelmet":
            if WITH_HELMET not in augmented_classes:
                continue

        if class_name == "WithoutHelmet":
            if WITHOUT_HELMET not in augmented_classes:
                continue

        # Filename
        new_name = (
            f"aug_{class_name.lower()}_"
            f"{generated:04d}_"
            f"{image_path.stem}"
        )

        image_output = (
            OUTPUT_IMAGES /
            f"{new_name}.jpg"
        )

        label_output = (
            OUTPUT_LABELS /
            f"{new_name}.txt"
        )

        # Save image
        cv2.imwrite(
            str(image_output),
            augmented_image,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        # Save transformed labels
        with open(label_output, "w") as f:

            for cls, bbox in zip(
                augmented_classes,
                augmented_boxes
            ):

                x, y, w, h = bbox

                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                f.write(
                    f"{int(cls)} "
                    f"{x:.6f} "
                    f"{y:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}\n"
                )

        generated += 1

        if generated % 50 == 0:

            print(
                f"  {class_name}: "
                f"{generated}/{needed}"
            )

    print(
        f"{class_name}: "
        f"{original_count + generated} images"
    )


# ============================================================
# OVERSAMPLE WITHHELMET
# ============================================================

augment_class(
    with_helmet_images,
    TARGET_WITH_HELMET,
    "WithHelmet"
)


# ============================================================
# OVERSAMPLE WITHOUTHELMET
# ============================================================

augment_class(
    without_helmet_images,
    TARGET_WITHOUT_HELMET,
    "WithoutHelmet"
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("OVERSAMPLING COMPLETED")
print("=" * 60)

print()
print("Output folder:")
print(OUTPUT_IMAGES)

print()
print("The original train/valid/test folders were not modified.")
print("All legitimate Plate labels were preserved.")