from pathlib import Path

def test_model_artifact_exists():
    p = Path("artifacts/models/model.pkl")
    assert p.exists(), "Run training first: `python src/get_data.py && python src/train.py`"
