from pathlib import Path


def test_model_artifact_exists():
    path = Path("artifacts/models/model.pkl")
    assert path.exists(), (
        "Model artifact missing. Run training: "
        "`python src/get_data.py && python src/train.py`"
    )
