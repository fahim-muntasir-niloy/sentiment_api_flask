import requests

url = 'http://localhost:5000/analyze'  # URL to make the request to
input_text = 'I am feeling not good'    # default req

payload = {'text': input_text}
response = requests.post(url, json=payload)

print(response.json())
