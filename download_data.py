import requests
import pandas as pd
import time

API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
RESOURCE_ID = "98e76922-ab13-474e-9bce-78942583cd0e"

all_data = []
offset = 0
limit = 1000
max_records = 20000

while True:
    url = f"https://api.data.gov.in/resource/{RESOURCE_ID}?api-key={API_KEY}&format=json&limit={limit}&offset={offset}"

    response = requests.get(url)

    if response.status_code != 200:
        print("API stopped with status:", response.status_code)
        break

    data = response.json()
    records = data.get("records", [])

    if not records:
        print("No more records.")
        break

    all_data.extend(records)
    offset += limit

    print(f"Downloaded {len(all_data)} records")

    if len(all_data) >= max_records:
        print("Reached 20,000 records. Stopping.")
        break

    time.sleep(2)

# Save only if data exists
if len(all_data) > 0:
    df = pd.DataFrame(all_data)
    df.drop_duplicates(inplace=True)
    df.to_csv("voicequality_data.csv", index=False)
    print("CSV Saved Successfully!")
else:
    print("No data to save.")