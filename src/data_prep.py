from pathlib import Path

import pandas as pd
import yaml


def main():
    """Minimal data-prep placeholder (no-op for this dataset)."""
    with open("src/config/config.yaml") as f:
        config = yaml.safe_load(f)

    csv_path = Path(config["data"]["output_csv"])
    df = pd.read_csv(csv_path)

    # Do any preprocessing here if needed. For now, just persist back.
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
