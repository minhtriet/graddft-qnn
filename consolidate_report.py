import json


def unique_dicts_by_key(dict_list, key):
    seen = set()
    unique_list = []

    for d in dict_list:
        if d[key] not in seen:
            unique_list.append(d)
            seen.add(d[key])

    return unique_list


with open("report.json") as f:
    original_json = json.load(f)
    consolidated_json = unique_dicts_by_key(original_json, "Date")

with open("report.json", "w") as f:
    json.dump(consolidated_json, f)
