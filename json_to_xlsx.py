import json
import pathlib

import pandas as pd

if pathlib.Path("report.json").exists():
    with open("report.json") as f:
        try:
            history_report = json.load(f)
        except json.decoder.JSONDecodeError:
            history_report = []
else:
    history_report = []
with open("report.json", "w") as f:
    json.dump(history_report, f)
pd.DataFrame(history_report).to_excel("report.xlsx")
