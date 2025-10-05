# tempo_hcho_l3_nyc_time.py
# 폴더의 TEMPO_HCHO_L3_V03_*.nc -> NYC BBOX 추출 -> CSV 병합
# time_utc 은 "파일명에 들어있는 시간"을 그대로 사용

import os, re
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob

# ===== 사용자 설정 =====
IN_DIR   = r""
OUT_CSV  = r".csv"
BBOX     = (-74.3, 40.4, -73.6, 41.0)  # NYC
REMOVE_NEGATIVE = True

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

# 파일명에서 시간 문자열 추출: YYYYMMDDThhmm 또는 YYYYMMDDThhmmss (뒤에 Z 있을 수도)
TS_PAT = re.compile(r".*?(\d{8}T\d{4,6})(?:Z|_)?", re.IGNORECASE)

def time_from_filename(fname: str) -> pd.Timestamp:
    m = TS_PAT.match(fname)
    if not m:
        raise ValueError(f"파일명에서 시간 패턴을 찾지 못함: {fname}")
    stamp = m.group(1)  # e.g., 20250618T1900 or 20250618T190023
    fmt = "%Y%m%dT%H%M%S" if len(stamp) == 15 else "%Y%m%dT%H%M"
    return pd.to_datetime(stamp, format=fmt, utc=True)

def extract_one(nc_path: str) -> pd.DataFrame:
    # 1) 루트에서 위경도 좌표
    root = xr.open_dataset(nc_path, engine="netcdf4", decode_cf=True, mask_and_scale=True)
    lat_name = "latitude" if "latitude" in root.coords else ("lat" if "lat" in root.coords else None)
    lon_name = "longitude" if "longitude" in root.coords else ("lon" if "lon" in root.coords else None)
    if not lat_name or not lon_name:
        raise RuntimeError(f"[{os.path.basename(nc_path)}] 위경도 좌표 없음: {list(root.coords)}")
    lat, lon = root[lat_name], root[lon_name]

    # 2) /product에서 값 읽기 (보통 'vertical_column')
    prod = xr.open_dataset(nc_path, group="product", engine="netcdf4", decode_cf=True, mask_and_scale=True)
    var = "vertical_column" if "vertical_column" in prod.data_vars else \
          next((v for v in prod.data_vars if ("column" in v.lower() or "hcho" in v.lower())), None)
    if var is None:
        raise RuntimeError(f"[{os.path.basename(nc_path)}] HCHO 변수 없음: {list(prod.data_vars)}")
    da = prod[var]

    # 3) 차원 이름 매핑(y/x → lat/lon)
    mapping = {}
    for d in da.dims:
        if da.sizes[d] == lat.size: mapping[d] = lat_name
        elif da.sizes[d] == lon.size: mapping[d] = lon_name
    da = da.rename(mapping).assign_coords({lat_name: lat, lon_name: lon})

    # 4) 결측/유효범위/음수 처리
    fill = da.encoding.get("_FillValue")
    if fill is not None: da = da.where(da != fill)
    for k in ("valid_min","valid_max"):
        v = da.attrs.get(k)
        if v is not None:
            da = da.where(da >= float(v)) if k=="valid_min" else da.where(da <= float(v))
    da = da.where(np.isfinite(da))
    if REMOVE_NEGATIVE:
        da = da.where(da > 0)

    # 5) NYC BBOX
    lon_min, lat_min, lon_max, lat_max = BBOX
    da = da.where(
        (da[lat_name] >= lat_min) & (da[lat_name] <= lat_max) &
        (da[lon_name] >= lon_min) & (da[lon_name] <= lon_max),
        drop=True
    )

    # 6) 표로 변환
    df = da.to_dataframe(name="hcho").reset_index().dropna(subset=["hcho"])

    # 7) 파일명 기반 시간 주입 (모든 행 동일 — 파일마다 다름)
    ts = time_from_filename(os.path.basename(nc_path))
    df["time_utc"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 부가 정보
    units = da.attrs.get("units")
    if units: df["units"] = units
    df["source_file"] = os.path.basename(nc_path)

    # 열 정리
    cols = ["time_utc", lat_name, lon_name, "hcho", "units", "source_file"]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    root.close(); prod.close()
    return df

def main():
    files = sorted(glob(os.path.join(IN_DIR, "*.nc")))
    if not files:
        raise FileNotFoundError(f".nc 파일이 없습니다: {IN_DIR}")

    out_list = []
    for p in files:
        try:
            out_list.append(extract_one(p))
            print(f"[OK] {os.path.basename(p)}")
        except Exception as e:
            print(f"[SKIP] {os.path.basename(p)} -> {e}")

    if not out_list:
        raise RuntimeError("처리 가능한 파일이 없습니다.")
    out = pd.concat(out_list, ignore_index=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ 완료: {OUT_CSV} (rows={len(out):,}, files={len(out_list)}/{len(files)})")

if __name__ == "__main__":
    main()
