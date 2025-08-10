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


# New Instructions added on Sunday
Checkout main code from repo: https://github.com/176akuma/ml_ops_assignment.git
Make required changes in repo and push changes to main branch.
Upon push, it will trigger pipeline. Status can be checked in GitHub repo. Go to repository in gitHub and Actions tab. You will see work flow runs. Latest one in running state. click on it to get detailed steps.
Above workflow train the model, test the code, build image, bring up container, check health and then brings down the container.
 Upon successful completion of above workflow run, it will post new image to docker hub - Path is akuma176/mlops-app. This can be checked in hub.docker.com with akuma176 login

Above image expects model in local to run. Hence before running, create model using below command
docker run --rm `
  -v ${PWD}\artifacts:/app/artifacts `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\mlruns:/app/mlruns `
  akuma176/mlops-app:latest `
  sh -c "python src/get_data.py && python src/train.py"

Once after building model run the image from hub
docker run --rm -p 5000:5000 `
  -v ${PWD}\artifacts:/app/artifacts `
  akuma176/mlops-app:latest

You can also use below commands
docker pull akuma176/mlops-app:latest
docker run --rm -p 5000:5000 akuma176/mlops-app:latest


```
