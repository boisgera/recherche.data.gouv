import json
import requests

URL = "https://entrepot.recherche.data.gouv.fr/api/search"
N = 1000 # max number of items on a single search API request

params = {"q": "*"}
response = requests.get(URL, params=params)
total_count = response.json()["data"]["total_count"]

start = 0
items = []

while start < total_count:
    print(start, total_count)
    params = {"q": "*", "start": start, "per_page": N}
    response = requests.get(URL, params=params)
    items.extend(response.json()["data"]["items"])
    start += N

with open("items.json", mode="wt", encoding="utf-8") as output:
    json.dump(items, output)