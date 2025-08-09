import requests

features = [2,3,4,5,6,7,8,9]
# data = {"features": 5.1, "features": 5.1, "features": 5.1, "features": 5.1, "features": 5.1, "features": 5.1, "features": 5.1, "features": 5.1,}
r = requests.post("http://127.0.0.1:5000/predict", json={"features": features})
print(r.json())