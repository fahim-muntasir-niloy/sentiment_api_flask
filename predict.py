import requests

url = 'http://localhost:5000/prediction'  # Update the URL if needed
input_text = 'I am feeling not good'

payload = {'text': input_text}
response = requests.post(url, json=payload)
prediction = response.json()['prediction']

print('Prediction:', prediction)
