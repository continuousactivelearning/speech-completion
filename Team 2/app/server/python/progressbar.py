import pandas as pd
import argparse


def time_to_seconds(t):
    """Convert MM:SS or HH:MM:SS string to seconds"""
    parts = list(map(int, t.strip().split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        return 0


def extract_start_end_sec(row):
    """Convert start/end times to seconds"""
    start_sec = time_to_seconds(row["start"])
    end_sec = time_to_seconds(row["end"])
    return pd.Series([start_sec, end_sec])


def main():
    parser = argparse.ArgumentParser(
        description="Add progress labels to clustered embeddings CSV"
    )
    parser.add_argument(
        "--input", required=True, help="Path to clustered_embeddings.csv"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save CSV with progress labels"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Number of bins for classification mode",
    )
    parser.add_argument(
        "--mode",
        choices=["classifier", "regression"],
        default="classifier",
        help="Label mode",
    )
    args = parser.parse_args()

    # Load and filter
    df = pd.read_csv(args.input)
    df = df[df["text"].notnull() & df["text"].str.strip().astype(bool)]
    print(f"Loaded {len(df)} rows")

    # Add start_sec, end_sec
    df[["start_sec", "end_sec"]] = df.apply(extract_start_end_sec, axis=1)

    progress_labels = []

    # Compute label per row
    for file_name, group in df.groupby("file"):
        total_duration = group["end_sec"].max()
        print(f"\nFile {file_name}: total duration {total_duration:.1f} sec")

        for _, row in group.iterrows():
            center_time = (row["start_sec"] + row["end_sec"]) / 2
            progress_frac = center_time / total_duration

            if args.mode == "classifier":
                class_size = 1.0 / args.num_classes
                progress_class = int(progress_frac / class_size)
                progress_class = min(progress_class, args.num_classes - 1)
                progress_labels.append(progress_class)

            elif args.mode == "regression":
                progress_labels.append(round(progress_frac, 6))  # keep float precision

    # Save
    df["progress_label"] = progress_labels
    df.to_csv(args.output, index=False)
    print(f"\nOK: saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
