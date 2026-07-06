import os
import io
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from scipy.interpolate import griddata
from scipy.ndimage import maximum_filter
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static files correctly
STATIC_DIR = os.path.join(os.getcwd(), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global cache
DF_RAW = None
TIME_LIST = []
CSV_URL = "https://data.weather.gov.hk/weatherAPI/hko_data/F3/Gridded_rainfall_nowcast_tc.csv"
MACAU_LON = 113.5685
MACAU_LAT = 22.1595
KM_PER_DEG = 111.32

def init_data():
    global DF_RAW, TIME_LIST
    resp = requests.get(CSV_URL, timeout=10)
    df = pd.read_csv(io.StringIO(resp.text), encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]
    lat_col = next(c for c in df.columns if "緯度" in c)
    lon_col = next(c for c in df.columns if "經度" in c)
    rain_col = next(c for c in df.columns if "雨量" in c)
    time_col = next((c for c in df.columns if "時間" in c and ("完結" in c or "結束" in c)), None)
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce')
    df = df.dropna()
    df[rain_col] = df[rain_col]
    df[time_col] = df[time_col].astype(str)
    DF_RAW = df
    TIME_LIST = sorted(df[time_col].unique())
    return lat_col, lon_col, rain_col, time_col

def sample_valid_points(df_sample, lon_col, lat_col, rain_col, min_dist_deg=0.02, min_rain=2.0):
    df_filter = df_sample[df_sample[rain_col] >= min_rain].copy()
    if df_filter.empty:
        return []
    selected = []
    lons = df_filter[lon_col].values
    lats = df_filter[lat_col].values
    rains = df_filter[rain_col].values
    for lon, lat, rain in zip(lons, lats, rains):
        too_close = False
        for slon, slat, _ in selected:
            dist = np.sqrt((lon - slon)**2 + (lat - slat)**2)
            if dist < min_dist_deg:
                too_close = True
                break
        if not too_close:
            selected.append([float(lon), float(lat), round(float(rain))])
    return selected

def process_timeslice(target_time: str):
    lat_col, lon_col, rain_col, time_col = init_data()
    df_frame = DF_RAW[DF_RAW[time_col] == target_time].copy()
    df_frame = df_frame[df_frame[rain_col] > 0]
    if df_frame.empty:
        raise Exception("No rainfall data")
    xi = np.linspace(df_frame[lon_col].min(), df_frame[lon_col].max(), 200)
    yi = np.linspace(df_frame[lat_col].min(), df_frame[lat_col].max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((df_frame[lon_col], df_frame[lat_col]), df_frame[rain_col], (Xi, Yi), method="cubic")
    Zi[Zi < 0.1] = np.nan
    grid_data = []
    for i in range(len(yi)):
        for j in range(len(xi)):
            val = float(Zi[i][j])
            if not np.isnan(val):
                grid_data.append([float(Xi[i][j]), float(Yi[i][j]), val])
    window_size = 8
    peak_min_threshold = 10.0
    max_filtered = maximum_filter(Zi, size=window_size)
    peak_mask = (Zi == max_filtered) & (~np.isnan(Zi))
    peak_lon_arr = Xi[peak_mask]
    peak_lat_arr = Yi[peak_mask]
    peak_val_arr = Zi[peak_mask]
    valid_idx = peak_val_arr >= peak_min_threshold
    peaks = []
    for lon, lat, val in zip(peak_lon_arr[valid_idx], peak_lat_arr[valid_idx], peak_val_arr[valid_idx]):
        peaks.append([float(lon), float(lat), round(float(val))])
    station_labels = sample_valid_points(df_frame, lon_col, lat_col)
    radar_rings = []
    ring_km = [10,25,50,100]
    ring_labels = ["10 km","25 km","50 km","100 km"]
    theta = np.linspace(0, 2*np.pi, 72)
    for km, lab in zip(ring_km, ring_labels):
        r_deg = km / KM_PER_DEG
        ring_points = []
        for t in theta:
            dlon = r_deg * np.cos(t)
            dlat = r_deg * np.sin(t)
            ring_points.append([MACAU_LAT + dlat, MACAU_LON + dlon])
        radar_rings.append({"label": lab, "points": ring_points})
    dt_obj = datetime.strptime(target_time, "%Y%m%d%H%M")
    time_display = dt_obj.strftime("%Y-%m-%d %H:%M")
    return {
        "time_raw": target_time,
        "time_display": time_display,
        "grid": grid_data,
        "peaks": peaks,
        "station_labels": station_labels,
        "radar_rings": radar_rings,
        "macau_center": [MACAU_LAT, MACAU_LON]
    }

# Homepage route (only / returns html)
@app.get("/")
def root():
    html_path = os.path.join(STATIC_DIR, "index.html")
    return FileResponse(html_path)

# All API routes return JSON only
@app.get("/api/refresh")
def api_refresh():
    init_data()
    return JSONResponse({"status": "ok", "count": len(TIME_LIST)})

@app.get("/api/timestamps")
def api_ts():
    ts_list = []
    for raw in TIME_LIST:
        dt = datetime.strptime(raw, "%Y%m%d%H%M")
        ts_list.append({"raw": raw, "display": dt.strftime("%Y-%m-%d %H:%M")})
    return JSONResponse({"timestamps": ts_list})

@app.get("/api/mapdata")
def api_mapdata(time: str = Query(...)):
    data = process_timeslice(time)
    return JSONResponse(data)

@app.on_event("startup")
def load_on_start():
    try:
        init_data()
        print("Data loaded successfully")
    except Exception as e:
        print(f"Startup warning: {e}")
