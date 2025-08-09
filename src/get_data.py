from sklearn.datasets import fetch_california_housing
import pandas as pd
from pathlib import Path
import yaml

with open("src/config/config.yaml") as f:
    config = yaml.safe_load(f)

out = Path(config["data"]["output_csv"])
out.parent.mkdir(parents=True, exist_ok=True)

data = fetch_california_housing(as_frame=True)
df = pd.concat([data.data, data.target.rename("MedHouseVal")], axis=1)
df.to_csv(out, index=False)
print(f"Saved {out} with shape {df.shape}")
