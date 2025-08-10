# MLOps Assignment (California Housing)
See the repo structure and instructions in the root of this ZIP. Start with:
```
docker build --no-cache -t mlops-app:local .

docker run --rm `
  -v ${PWD}\artifacts:/app/artifacts `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\mlruns:/app/mlruns `
  mlops-app:local `
  bash -lc "python src/get_data.py && python src/train.py"


docker run --rm -p 5000:5000 `
  -v ${PWD}\artifacts:/app/artifacts `
  mlops-app:local


cat artifacts/logs/app.log


sqlite3 artifacts/db/preds.sqlite "SELECT * FROM predictions LIMIT 5;"

curl http://localhost:5000/metrics | Select-String "predictions_total"

docker run --rm `
  -p 5001:5001 `
  -v ${PWD}\mlruns:/app/mlruns `
  mlops-app:local `
  sh -c "mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5001"



```
