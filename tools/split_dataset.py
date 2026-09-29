import json
from pathlib import Path

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
SPLIT_DIR = PROJECT_ROOT / "dataset" / "splits"

CLASSES = [
    "circle",
    "ellipse",
    "triangle",
    "square",
    "rectangle",
    "pentagon",
    "hexagon",
]


def load_dataset():
    samples = []

    for label in CLASSES:
        class_dir = PROCESSED_DIR / label

        if not class_dir.exists():
            continue

        for file_path in sorted(class_dir.glob("sample_*.json")):
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            samples.append(data)

    return samples


def save_split(samples, name):
    output_dir = SPLIT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples):
        output_path = output_dir / f"sample_{index:04d}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(sample, f)


def main():
    samples = load_dataset()

    if not samples:
        raise RuntimeError("No processed samples found.")

    X = samples
    y = [sample["label"] for sample in samples]

    # First: 70% train, 30% temporary
    train, temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # Split temporary 50/50 -> 15% validation, 15% test
    validation, test, _, _ = train_test_split(
        temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    # Remove previous split
    if SPLIT_DIR.exists():
        for file in SPLIT_DIR.rglob("*.json"):
            file.unlink()

    save_split(train, "train")
    save_split(validation, "validation")
    save_split(test, "test")

    print("=" * 50)
    print("DATASET SPLIT")
    print("=" * 50)
    print(f"Total:      {len(samples)}")
    print(f"Training:   {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")
    print("=" * 50)


if __name__ == "__main__":
    main()