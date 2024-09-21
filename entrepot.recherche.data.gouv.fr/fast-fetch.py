import os
import json

from joblib import Parallel, delayed
import requests

URL = "https://entrepot.recherche.data.gouv.fr/api/search"
N = 1000 # max number of items on a single search API request

params = {"q": "*"}
response = requests.get(URL, params=params)
total_count = response.json()["data"]["total_count"]

def get_chunck(start):
    params = {"q": "*", "start": start, "per_page": N}
    response = requests.get(URL, params=params)
    return response.json()["data"]["items"]


parallel = Parallel(n_jobs=os.cpu_count(), return_as="generator")

starts = range(0, total_count, N)

output_generator = parallel(delayed(get_chunck)(start) for start in starts)

items = []
for i, datum in zip(starts, output_generator):
    items.extend(datum)

with open("items.json", mode="wt", encoding="utf-8") as output:
    json.dump(items, output)

print("Done!")

