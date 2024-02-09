import json

with open("items.json", mode="rt", encoding="utf-8") as input:
    items = json.load(input)

types = {}

for item in items:
    t = item["type"]
    types.setdefault(t, 0)
    types[t] += 1

print(types)

files = [item for item in items if item["type"] == "file"]

for file in files[:10]:
    print(file)

size = sum(file["size_in_bytes"] for file in files)

print(f"{size/1_000_000_000 = } GB")

file_type = set()
file_content_type = set()

for file in files:
    file_type.add(file["file_type"])
    file_content_type.add(file["file_content_type"])

file_type = sorted(list(file_type))
file_content_type = sorted(list(file_content_type))

print("File type:")
for ft in file_type:
    print(f"  - {ft}")

print("File content type:")
for fct in file_content_type:
    print(f"  - {fct}")

file_types = {}
for file in files:
    ft = file["file_type"]
    mt = file["file_content_type"]
    file_types.setdefault((ft, mt), 0)
    file_types[(ft, mt)] += 1

file_types = list(file_types.items())
file_types = sorted(file_types, key=lambda item: -item[1])

for key, value in file_types:
    print(f"{key}: {value}")