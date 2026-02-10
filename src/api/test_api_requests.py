import requests

url = "http://127.0.0.1:8000/predict"

samples = [
    "I was charged twice and my invoice is wrong",
    "My delivery is late and tracking is not working",
    "I want a refund, I returned the product",
    "The app keeps crashing and freezing",
    "My account is locked and I cannot reset my password",
    "Do you offer discounts for new customers?"
]

for text in samples:
    response = requests.post(url, json={"text": text})
    print("INPUT:", text)
    print("OUTPUT:", response.json())
    print("-" * 50)
