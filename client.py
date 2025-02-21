import requests

url = "http://localhost:8000/generate/"
payload = {"prompt": "Explain AI's impact on sales.", "max_length": 100}

response = requests.post(url, json=payload)
print(response.json())

