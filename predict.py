import requests

url = 'http://localhost:5000/prediction'  # URL to make the request to
input_text = 'I am feeling not good'    # default req

payload = {'text': input_text}
response = requests.post(url, json=payload)
prediction = response.json()['prediction']

print('Sentiment:', prediction)
