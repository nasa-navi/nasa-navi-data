# 측정 시각을 nc 파일명 내 시각으로 변환하는 코드
# 폴더의 TEMPO_NO2_L3_V03_*.nc -> NYC BBOX 추출 -> CSV 병합
# time_utc 은 "파일명에 들어있는 시간"을 그대로 사용

import os, re
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from typing import Optional, Dict

# ===== 사용자 설정 =====
IN_DIR   = r""
OUT_CSV  = r""
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

def pick_var_by_candidates(dvars: Dict[str, xr.DataArray], candidates):
    """data_vars에서 후보명을 순서대로 찾아 첫 매칭을 반환. 없으면 None"""
    lower_map = {k.lower(): k for k in dvars.keys()}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def find_no2_var(prod: xr.Dataset) -> str:
    # 일반적으로 TEMPO NO2 L3는 vertical_column_troposphere 사용
    candidates = [
        "vertical_column_troposphere",
        "vertical_column",
        "no2_vertical_column",
        "no2_column",
    ]
    name = pick_var_by_candidates(prod.data_vars, candidates)
    if name:
        return name
    # fallback: 'no2' 또는 'column' 포함 변수 중 유력한 것 선택
    for v in prod.data_vars:
        vlow = v.lower()
        if "no2" in vlow and ("column" in vlow or "vertical" in vlow):
            return v
    for v in prod.data_vars:
        if "column" in v.lower():
            return v
    raise RuntimeError(f"NO2 변수 자동 탐색 실패: {list(prod.data_vars)}")

def find_cloud_fraction_var(prod: xr.Dataset) -> Optional[str]:
    # 다양한 이름 가능성 대비
    candidates = [
        "cloud_fraction",
        "scene_cloud_fraction",
        "cloudfrac",
        "cloud_fraction_troposphere",
        "cloud_fraction_total",
        "cloud_fraction_scene",
    ]
    name = pick_var_by_candidates(prod.data_vars, candidates)
    if name:
        return name
    # 폭넓은 휴리스틱: 'cloud'와 'fraction' 모두 포함
    for v in prod.data_vars:
        vlow = v.lower()
        if ("cloud" in vlow) and ("fraction" in vlow):
            return v
    return None

def align_and_clean(da: xr.DataArray, lat, lon, lat_name, lon_name) -> xr.DataArray:
    # 차원 이름 매핑(y/x → lat/lon)
    mapping = {}
    for d in da.dims:
        if da.sizes[d] == lat.size:
            mapping[d] = lat_name
        elif da.sizes[d] == lon.size:
            mapping[d] = lon_name
    da = da.rename(mapping).assign_coords({lat_name: lat, lon_name: lon})

    # 결측/유효범위/음수 처리
    fill = da.encoding.get("_FillValue")
    if fill is not None:
        da = da.where(da != fill)

    valid_min = da.attrs.get("valid_min")
    valid_max = da.attrs.get("valid_max")
    if valid_min is not None:
        da = da.where(da >= float(valid_min))
    if valid_max is not None:
        da = da.where(da <= float(valid_max))

    da = da.where(np.isfinite(da))
    return da

def apply_bbox(da: xr.DataArray, lat_name: str, lon_name: str, bbox) -> xr.DataArray:
    if bbox is None:
        return da
    lon_min, lat_min, lon_max, lat_max = bbox
    return da.where(
        (da[lat_name] >= lat_min) & (da[lat_name] <= lat_max) &
        (da[lon_name] >= lon_min) & (da[lon_name] <= lon_max),
        drop=True
    )

def extract_one(nc_path: str) -> pd.DataFrame:
    # 1) 루트에서 위경도 좌표
    root = xr.open_dataset(nc_path, engine="netcdf4", decode_cf=True, mask_and_scale=True)
    lat_name = "latitude" if "latitude" in root.coords else ("lat" if "lat" in root.coords else None)
    lon_name = "longitude" if "longitude" in root.coords else ("lon" if "lon" in root.coords else None)
    if not lat_name or not lon_name:
        raise RuntimeError(f"[{os.path.basename(nc_path)}] 위경도 좌표 없음: {list(root.coords)}")
    lat, lon = root[lat_name], root[lon_name]

    # 2) /product에서 값 읽기 (NO2 + cloud fraction)
    prod = xr.open_dataset(nc_path, group="product", engine="netcdf4", decode_cf=True, mask_and_scale=True)

    # --- NO2 본변수 ---
    no2_var_name = find_no2_var(prod)
    no2_da = align_and_clean(prod[no2_var_name], lat, lon, lat_name, lon_name)
    if REMOVE_NEGATIVE:
        no2_da = no2_da.where(no2_da > 0)

    # --- Cloud fraction(있으면) ---
    cf_name = find_cloud_fraction_var(prod)
    cf_da = None
    if cf_name is not None:
        cf_da = align_and_clean(prod[cf_name], lat, lon, lat_name, lon_name)
        # 일반적으로 0~1 범위. 유효범위가 있으면 위에서 정리됨.

    # 3) NYC BBOX
    no2_da = apply_bbox(no2_da, lat_name, lon_name, BBOX)
    if cf_da is not None:
        cf_da = apply_bbox(cf_da, lat_name, lon_name, BBOX)

    # 4) 표로 변환
    df_no2 = no2_da.to_dataframe(name="no2").reset_index().dropna(subset=["no2"])
    # 🔧 L3에서 남는 보조 차원 'time'이 붙으면 제거 (안전)
    if "time" in df_no2.columns:
        df_no2 = df_no2.drop(columns=["time"], errors="ignore")

    if cf_da is not None:
        df_cf = cf_da.to_dataframe(name="cloud_fraction").reset_index()
        if "time" in df_cf.columns:
            df_cf = df_cf.drop(columns=["time"], errors="ignore")
        # lat/lon 기준 병합
        df = pd.merge(df_no2, df_cf[[lat_name, lon_name, "cloud_fraction"]],
                      on=[lat_name, lon_name], how="left")
    else:
        df = df_no2

    # 5) 파일명 기반 시간 주입 (모든 행 동일 — 파일마다 다름)
    ts = time_from_filename(os.path.basename(nc_path))
    df["time_utc"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 부가 정보
    units = no2_da.attrs.get("units")
    if units:
        df["no2_units"] = units
    if cf_da is not None and cf_da.attrs.get("units"):
        df["cloud_fraction_units"] = cf_da.attrs.get("units")

    df["source_file"] = os.path.basename(nc_path)

    # 열 정리: time, lat, lon, no2, cloud_fraction(옵션), units, source_file 순
    base_cols = ["time_utc", lat_name, lon_name, "no2"]
    if "cloud_fraction" in df.columns:
        base_cols.append("cloud_fraction")
    if "no2_units" in df.columns:
        base_cols.append("no2_units")
    if "cloud_fraction_units" in df.columns:
        base_cols.append("cloud_fraction_units")
    base_cols.append("source_file")

    df = df[base_cols + [c for c in df.columns if c not in base_cols]]
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
    print(f"\n 완료: {OUT_CSV} (rows={len(out):,}, files={len(out_list)}/{len(files)})")

if __name__ == "__main__":
    main()
