# 폴더의 TEMPO_O3TOT_L3_V03 *.nc → NYC BBOX 크롭 → 필요한 변수만 CSV 병합
# ※ 모든 행의 'time'은 해당 nc "파일명"의 시각으로 덮어씀 (열 손상 없음)

import os, re
import pandas as pd
import xarray as xr

# ===== 사용자 설정 =====
IN_DIR  = r""
OUT_CSV = r""
BBOX    = (-74.3, 40.4, -73.6, 41.0)  # NYC (lon_min, lat_min, lon_max, lat_max). 전체면 None

# TEMPO 파일명 예: TEMPO_O3TOT_L3_V03_20250601T103345Z_S001.nc
TS_PAT = re.compile(r"_(\d{8}T\d{6})Z", re.IGNORECASE)

# ===== O3 변수 매핑 규칙 =====
MAIN_O3_CANDS = [
    "column_amount_o3",
    "total_ozone_column", "ozone_total_column", "ozone_column_total",
    "o3_total_column", "o3_column_total", "tco", "ozone_total"
]
PRECISION_CANDS = [
    "total_ozone_column_precision", "total_ozone_column_uncertainty",
    "ozone_total_column_precision", "ozone_total_column_uncertainty",
    "o3_total_column_precision", "o3_total_column_uncertainty",
]
QA_CANDS  = ["qa_value", "quality_flag", "quality_value", "qa"]
ECF_CANDS = ["effective_cloud_fraction", "fc", "cloud_fraction", "cloud_frac", "cloud_radiance_fraction"]
RCF_CANDS = ["radiative_cloud_fraction", "radiative_cloud_frac"]
OCP_CANDS = ["cloud_optical_centroid_pressure", "ocp"]
SZA_CANDS = ["solar_zenith_angle", "sza"]
VZA_CANDS = ["viewing_zenith_angle", "vza"]

# ===== 유틸 =====
def open_root_and_product(path):
    for eng in ("netcdf4", "h5netcdf"):
        try:
            root = xr.open_dataset(path, engine=eng)
            try:
                prod = xr.open_dataset(path, engine=eng, group="product")
            except Exception:
                prod = root
            return root, prod
        except Exception:
            continue
    raise RuntimeError("netCDF 파일을 열 수 없습니다 (netcdf4/h5netcdf 확인).")

def find_lat_lon_any(root, prod):
    cand_lat = ["latitude", "lat", "y"]
    cand_lon = ["longitude", "lon", "x"]
    def _has(ds, name): return (name in ds.coords) or (name in ds.variables)
    lat = next((c for c in cand_lat if _has(root, c) or _has(prod, c)), None)
    lon = next((c for c in cand_lon if _has(root, c) or _has(prod, c)), None)
    if lat is None or lon is None:
        raise ValueError("위/경도 좌표(latitude/longitude)를 찾을 수 없습니다.")
    return lat, lon

def first_match(ds, cands):
    lowers = {k.lower(): k for k in ds.data_vars}
    for nick in cands:
        for var in ds.data_vars:
            if nick.lower() in var.lower():  # 부분 포함
                return var
        if nick.lower() in lowers:          # 정확 일치
            return lowers[nick.lower()]
    return None

def pick_main_o3(ds):
    v = first_match(ds, MAIN_O3_CANDS)
    if v: return v
    for var in ds.data_vars:
        lv = var.lower()
        if "ozone" in lv and "column" in lv: return var
    for var in ds.data_vars:
        lv = var.lower()
        if "o3" in lv and "column" in lv: return var
    for var in ds.data_vars:
        if "ozone" in var.lower(): return var
    return None

def ensure_coords(da, latname, lonname, root, prod):
    if latname in da.coords and lonname in da.coords:
        return da
    lat = (root[latname] if (latname in root.variables or latname in root.coords) else prod[latname])
    lon = (root[lonname] if (lonname in root.variables or lonname in root.coords) else prod[lonname])
    return da.assign_coords({latname: lat, lonname: lon})

def apply_bbox(da, latname, lonname, bbox):
    if bbox is None:
        return da
    lon_min, lat_min, lon_max, lat_max = bbox
    return da.where(
        (da[latname] >= lat_min) & (da[latname] <= lat_max) &
        (da[lonname] >= lon_min) & (da[lonname] <= lon_max),
        drop=True
    )

def add_optional(prod, df, out_name, cands, latname, lonname, root):
    var = first_match(prod, cands)
    if var is None:
        return df
    da = prod[var]
    da = ensure_coords(da, latname, lonname, root, prod)
    da = apply_bbox(da, latname, lonname, BBOX)
    dfx = da.to_dataframe(name=out_name).reset_index()
    on_cols = [c for c in ["time", latname, lonname] if c in dfx.columns and c in df.columns]
    if len(on_cols) < 2:
        on_cols = [c for c in [latname, lonname] if c in dfx.columns and c in df.columns]
    return df.merge(dfx, on=on_cols, how="left")

def time_from_filename(fname: str) -> str:
    m = TS_PAT.search(fname)
    if not m:
        return None
    tstr = m.group(1)  # YYYYMMDDTHHMMSS
    ts = pd.to_datetime(tstr, format="%Y%m%dT%H%M%S", utc=True)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

# ===== 메인 =====
def main():
    files = [f for f in sorted(os.listdir(IN_DIR)) if f.lower().endswith(".nc")]
    if not files:
        raise FileNotFoundError(f".nc 파일이 없습니다: {IN_DIR}")

    all_rows = []

    for fname in files:
        path = os.path.join(IN_DIR, fname)
        print(f"[처리] {fname}")
        try:
            root, prod = open_root_and_product(path)
            latname, lonname = find_lat_lon_any(root, prod)

            main_var = pick_main_o3(prod)
            if main_var is None:
                print(" - 총오존 변수 탐지 실패 → 건너뜀")
                root.close();  prod is not root and prod.close()
                continue

            # 메인 변수
            da_main = prod[main_var]
            da_main = ensure_coords(da_main, latname, lonname, root, prod)
            da_main = apply_bbox(da_main, latname, lonname, BBOX)

            df = da_main.to_dataframe(name="total_ozone_column").reset_index()
            df = df.dropna(subset=["total_ozone_column"])

            # 보조 변수(있을 때만)
            df = add_optional(prod, df, "total_ozone_column_precision", PRECISION_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "effective_cloud_fraction", ECF_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "radiative_cloud_fraction", RCF_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "cloud_optical_centroid_pressure", OCP_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "solar_zenith_angle", SZA_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "viewing_zenith_angle", VZA_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "qa_value", QA_CANDS, latname, lonname, root)

            # === 핵심: time을 "파일명"에서 추출해 덮어쓰기 ===
            time_iso = time_from_filename(fname)
            if time_iso is None:
                raise RuntimeError(f"파일명에서 시간 패턴을 찾지 못했습니다: {fname}")
            df["time"] = time_iso  # 모든 행 동일(파일별 시각)

            # 메타(다른 열은 그대로 유지)
            df["source_file"]  = fname
            df["units"]        = da_main.attrs.get("units", "")  # 보통 "DU"
            df["product_kind"] = "o3"

            # 열 순서 정리(있는 것만)
            base = ["time", latname, lonname, "total_ozone_column"]
            extras = [c for c in [
                "total_ozone_column_precision",
                "effective_cloud_fraction",
                "radiative_cloud_fraction",
                "cloud_optical_centroid_pressure",
                "solar_zenith_angle",
                "viewing_zenith_angle",
                "qa_value",
            ] if c in df.columns]
            tail = ["source_file", "units", "product_kind"]
            df = df[[c for c in base + extras + tail if c in df.columns]]

            all_rows.append(df)

            root.close()
            if prod is not root: prod.close()

        except Exception as e:
            print(f" - 오류: {e}")
            continue

    out = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n✅ 완료: {OUT_CSV} (rows={len(out):,}, files={len(all_rows)}/{len(files)})")

if __name__ == "__main__":
    main()
