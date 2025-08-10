# One-Page Summary

- **Dataset**: California Housing (sklearn). Stored as CSV; optionally tracked by DVC.
- **Models**: Linear Regression & Random Forest; RMSE used for selection.
- **Tracking**: MLflow logs params/metrics/artifacts in `./mlruns`.
- **API**: FastAPI (`/predict`, `/health`, `/metrics`). Validation via Pydantic. 
- **Monitoring**: Prometheus `predictions_total`; file & SQLite logs.
- **Docker**: Python 3.11 slim; exposes 5000.
- **CI/CD**: GitHub Actions—lint, train smoke, tests, Docker build & push (if secrets).
- **Demo**: Train, run API, call `/predict`, show logs & DB, show GH Actions & Docker Hub.
