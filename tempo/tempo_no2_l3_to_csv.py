import os
import pandas as pd
import xarray as xr

IN_DIR  = r""   # NO2 L3 .nc 폴더
OUT_DIR = r""
BBOX    = (-74.3, 40.4, -73.6, 41.0)
OUT_CSV = "no2_L3_merged_NYC_with_fraction.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# ----- 탐지 규칙 -----
MAIN_CANDIDATES = [
    "vertical_column_troposphere",                 # 최우선
    "tropospheric_vertical_column",                # 변형
    "no2_tropospheric_vertical_column",
    "no2_vertical_column_troposphere",
    "no2_column_troposphere"
]

EXTRA_CANDIDATES = {
    # cloud
    "cloud_fraction": [
        "cloud_fraction", "effective_cloud_fraction", "cloud_radiance_fraction", "cloud_frac"
    ],
    # precision/uncertainty
    "vertical_column_troposphere_precision": [
        "vertical_column_troposphere_precision", "tropospheric_vertical_column_precision",
        "no2_tropospheric_vertical_column_precision", "precision_trop"
    ],
    # QA
    "qa_value": [
        "qa_value", "qa", "quality_flag", "quality", "quality_value"
    ],
    # (예시) 공기질 모델 보정용 AMF 등
    "air_mass_factor_troposphere": [
        "air_mass_factor_troposphere", "amf_troposphere", "tropospheric_amf"
    ],
}

def first_match(ds, names):
    low = {k.lower(): k for k in ds.data_vars}
    for nick in names:
        for var in ds.data_vars:
            if nick.lower() in var.lower():
                return var
        if nick.lower() in low:
            return low[nick.lower()]
    return None

def pick_main_no2(ds):
    # 우선 고정 이름 후보로, 없으면 'no2'+'column' 포함 2D
    v = first_match(ds, MAIN_CANDIDATES)
    if v: return v
    for var in ds.data_vars:
        vl = var.lower()
        if "no2" in vl and "column" in vl:
            return var
    # 그래도 없으면 'no2' 포함 2D
    for var in ds.data_vars:
        if "no2" in var.lower():
            return var
    # 마지막 fallback: 'column' 포함
    for var in ds.data_vars:
        if "column" in var.lower():
            return var
    return None

all_rows = []

for fname in sorted(os.listdir(IN_DIR)):
    if not fname.endswith(".nc"): 
        continue
    path = os.path.join(IN_DIR, fname)
    print(f"\n[읽는 중] {fname}")

    try:
        root = xr.open_dataset(path, engine="netcdf4")
        # 시간 메타(파일 메타 우선)
        s0 = root.attrs.get("time_coverage_start_since_epoch")
        s1 = root.attrs.get("time_coverage_end_since_epoch")
        if s0 is not None and s1 is not None:
            t0 = pd.to_datetime(float(s0), unit="s", utc=True)
            t1 = pd.to_datetime(float(s1), unit="s", utc=True)
            tm = t0 + (t1 - t0)/2
        else:
            tm = pd.to_datetime(root["time"].values, utc=True)[0]
            t0 = tm; t1 = tm

        lat = root["latitude"]; lon = root["longitude"]
        prod = xr.open_dataset(path, group="product", engine="netcdf4")

        main_var = pick_main_no2(prod)
        if main_var is None:
            print(f"NO2 변수 탐지 실패 → 건너뜀 ({fname})")
            root.close(); prod.close()
            continue

        # 좌표 주입(없으면)
        da_main = prod[main_var]
        if "latitude" not in da_main.coords or "longitude" not in da_main.coords:
            da_main = da_main.assign_coords(latitude=lat, longitude=lon)

        # BBOX
        lon_min, lat_min, lon_max, lat_max = BBOX
        da_main = da_main.where(
            (da_main["latitude"] >= lat_min) & (da_main["latitude"] <= lat_max) &
            (da_main["longitude"] >= lon_min) & (da_main["longitude"] <= lon_max),
            drop=True
        )

        # 메인 DF
        df = da_main.to_dataframe(name="vertical_column_troposphere").reset_index()
        df = df.dropna(subset=["vertical_column_troposphere"])

        # ----- 보조변수 병합 -----
        for out_name, candidates in EXTRA_CANDIDATES.items():
            var = first_match(prod, candidates)
            if var is None:
                continue
            da = prod[var]
            if "latitude" not in da.coords or "longitude" not in da.coords:
                da = da.assign_coords(latitude=lat, longitude=lon)
            da = da.where(
                (da["latitude"] >= lat_min) & (da["latitude"] <= lat_max) &
                (da["longitude"] >= lon_min) & (da["longitude"] <= lon_max),
                drop=True
            )
            dfx = da.to_dataframe(name=out_name).reset_index()
            # 같은 좌표(time,lat,lon) 기준 좌측 병합
            on_cols = [c for c in ["time","latitude","longitude"] if c in dfx.columns and c in df.columns]
            if len(on_cols) < 2:  # 좌표가 time 없이 lat/lon만인 경우 처리
                on_cols = [c for c in ["latitude","longitude"] if c in dfx.columns and c in df.columns]
            df = df.merge(dfx, on=on_cols, how="left")

        # 메타 컬럼
        df["time_start_utc"] = t0.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        df["time_end_utc"]   = t1.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        df["time_mid_utc"]   = tm.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        df["source_file"]    = fname
        df["units"]          = da_main.attrs.get("units", "")
        df["product_kind"]   = "no2"

        # 최종 컬럼 순서(있으면 포함)
        base = ["time","latitude","longitude","vertical_column_troposphere"]
        extras = [c for c in ["cloud_fraction",
                              "vertical_column_troposphere_precision",
                              "qa_value",
                              "air_mass_factor_troposphere"] if c in df.columns]
        tail = ["time_start_utc","time_end_utc","time_mid_utc","source_file","units","product_kind"]
        df = df[[c for c in base + extras + tail if c in df.columns]]

        all_rows.append(df)

        root.close(); prod.close()

    except Exception as e:
        print(f" 오류 ({fname}): {e}")
        continue

if all_rows:
    out = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(OUT_DIR, OUT_CSV)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n 완료: {len(out):,}개 행 → {out_path}")
else:
    print(" 변환된 데이터 없음")
