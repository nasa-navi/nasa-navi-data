import os
import numpy as np
import pandas as pd
import xarray as xr

# ===== 사용자 설정 =====
IN_DIR = r""   # nc 파일이 있는 폴더
OUT_DIR = r""
BBOX = (-74.3, 40.4, -73.6, 41.0)  # 뉴욕 근방
os.makedirs(OUT_DIR, exist_ok=True)

# ===== CSV 합치기 준비 =====
all_records = []

# ===== 모든 파일 순회 =====
for fname in sorted(os.listdir(IN_DIR)):
    if not fname.endswith(".nc"):
        continue
    path = os.path.join(IN_DIR, fname)
    print(f"\n[읽는 중] {fname}")

    try:
        # ---- 루트 그룹에서 시간 메타 추출 ----
        root = xr.open_dataset(path, engine="netcdf4")
        start_s = root.attrs.get("time_coverage_start_since_epoch")
        end_s   = root.attrs.get("time_coverage_end_since_epoch")

        if start_s and end_s:
            t_start = pd.to_datetime(float(start_s), unit="s", utc=True)
            t_end   = pd.to_datetime(float(end_s),   unit="s", utc=True)
            t_mid   = t_start + (t_end - t_start)/2
        else:
            t_mid = pd.to_datetime(root["time"].values, utc=True)[0]
            t_start = t_mid
            t_end = t_mid

        lat = root["latitude"]
        lon = root["longitude"]

        # ---- /product 그룹에서 변수 추출 ----
        prod = xr.open_dataset(path, group="product", engine="netcdf4")
        var = None
        for v in prod.data_vars:
            if "column" in v.lower() or "hcho" in v.lower():
                var = v
                break
        if var is None:
            print(f"⚠️ HCHO 변수 없음 → 건너뜀 ({fname})")
            continue

        da = prod[var]

        # 좌표 보정
        da = da.assign_coords(latitude=lat, longitude=lon)

        # NYC 범위만 선택
        lon_min, lat_min, lon_max, lat_max = BBOX
        da = da.where(
            (da["latitude"] >= lat_min) & (da["latitude"] <= lat_max) &
            (da["longitude"] >= lon_min) & (da["longitude"] <= lon_max),
            drop=True
        )

        df = da.to_dataframe(name="hcho").reset_index().dropna(subset=["hcho"])
        df["time_start_utc"] = t_start.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        df["time_end_utc"]   = t_end.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        df["time_mid_utc"]   = t_mid.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        df["source_file"] = fname

        units = da.attrs.get("units", None)
        if units:
            df["units"] = units

        all_records.append(df)

        root.close()
        prod.close()

    except Exception as e:
        print(f"❌ 오류 발생 ({fname}): {e}")
        continue

# ===== 전체 병합 후 저장 =====
if all_records:
    df_all = pd.concat(all_records, ignore_index=True)
    out_csv = os.path.join(OUT_DIR, "hcho_L3_2025_06_NYC.csv")
    df_all.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n✅ 완료: {len(df_all):,}개 행 → {out_csv}")
else:
    print("❌ 변환된 데이터 없음")