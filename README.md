# mlops-pipeline
docker build -t ml_ops_assignment_image .


docker run --rm --name mlops -p 5000:5000 ml_ops_assignment_image