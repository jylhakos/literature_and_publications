# client.py

# importing the requests library
import requests
# defining the api-endpoint
API_ENDPOINT = "http://0.0.0.0:8888/invocations"

# data to be sent to api
param = {
  "inputs": [[[ 0.074], [-0.003], [-0.08] ,[ -0.157], [-0.235], [-0.312]]]
}

# sending post request and saving response as a json object
response = requests.post(url = API_ENDPOINT, json = param)
result = response.json()
# extracting result
print(result[0][0])