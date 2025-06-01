import requests
import pandas as pd
from datetime import datetime, timedelta

# API bilgilerimiz
api_key = ""
api_host = ""

headers = {
    "X-RapidAPI-Key": api_key,
    "X-RapidAPI-Host": api_host
}

station_id = "17060"  ##İstanbul Atatürk Havalimanı
start_date = datetime(1996, 1, 1)
end_date = datetime(2023, 4, 30)  # Yaklaşık 10.000 gün

all_data = pd.DataFrame()

# Yıl yıl veri çektik
current = start_date
while current < end_date:
    next_year = current + timedelta(days=365)
    if next_year > end_date:
        next_year = end_date

    url = "https://meteostat.p.rapidapi.com/stations/daily"
    params = {
        "station": station_id,
        "start": current.strftime('%Y-%m-%d'),
        "end": next_year.strftime('%Y-%m-%d')
    }

    print(f"Veri alınıyor: {params['start']} - {params['end']}")
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json().get("data", [])
        df = pd.DataFrame(data)
        all_data = pd.concat([all_data, df], ignore_index=True)
    else:
        print("Hata:", response.status_code, response.text)

    current = next_year + timedelta(days=1)

# CSV'ye kaydediyor
all_data.to_csv("istanbul_10000_gun.csv", index=False)
print(" Toplam veri sayısı:", len(all_data), "- Dosya kaydedildi.")
