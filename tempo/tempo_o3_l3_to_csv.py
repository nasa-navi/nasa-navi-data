# tempo_o3_l3_to_csv.py
# 폴더의 TEMPO_O3TOT_L3_V03 *.nc → NYC BBOX로 크롭 → 필요한 변수만 CSV 병합

# tempo_o3_l3_to_csv_fix.py
# TEMPO_O3TOT_L3_V03 *.nc → NYC BBOX 크롭 → 앱에 필요한 컬럼만 CSV 병합
# - 위/경도는 root 그룹에서 가져와 product 변수에 주입
# - O3 전용 변수 후보(총오존/클라우드/기하/QA)로 유연 탐지

import os, re
import pandas as pd
import xarray as xr

# ===== 사용자 설정 =====
IN_DIR  = r""
OUT_DIR = r""
BBOX    = (-74.3, 40.4, -73.6, 41.0)   # NYC (lon_min, lat_min, lon_max, lat_max). 전체면 None
OUT_CSV = "o3_L3_merged_NYC_min.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== O3 변수 매핑 규칙 =====
# 총오존(메인)
MAIN_O3_CANDS = [
    "column_amount_o3",                 # TEMPO O3 L3에서 가장 흔함 (단위 DU)
    "total_ozone_column", "ozone_total_column", "ozone_column_total",
    "o3_total_column", "o3_column_total", "tco", "ozone_total"
]
# 불확도/정밀도
PRECISION_CANDS = [
    "total_ozone_column_precision", "total_ozone_column_uncertainty",
    "ozone_total_column_precision", "ozone_total_column_uncertainty",
    "o3_total_column_precision", "o3_total_column_uncertainty",
    "precision_total_ozone", "uncertainty_total_ozone"
]
# 품질
QA_CANDS = ["qa_value", "quality_flag", "quality_value", "qa"]
# 클라우드
ECF_CANDS = ["effective_cloud_fraction", "fc", "cloud_fraction", "cloud_frac", "cloud_radiance_fraction"]
RCF_CANDS = ["radiative_cloud_fraction", "radiative_cloud_frac"]
OCP_CANDS = ["cloud_optical_centroid_pressure", "ocp"]
# 관측 기하(있으면)
SZA_CANDS = ["solar_zenith_angle", "sza"]
VZA_CANDS = ["viewing_zenith_angle", "vza"]

# ===== 유틸 =====
def open_root_and_product(path):
    # netcdf4가 안 되면 h5netcdf로도 열 수 있게 시도
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
    raise RuntimeError("이 파일을 netCDF로 열 수 없습니다. netCDF4 또는 h5netcdf 설치를 확인하세요.")

def find_lat_lon_any(root, prod):
    # latitude/longitude는 보통 root에만 있음. 둘 다 합쳐서 검색.
    cand_lat = ["latitude", "lat", "y"]
    cand_lon = ["longitude", "lon", "x"]

    def _has(ds, name):
        return (name in ds.coords) or (name in ds.variables)

    lat = next((c for c in cand_lat if _has(root, c) or _has(prod, c)), None)
    lon = next((c for c in cand_lon if _has(root, c) or _has(prod, c)), None)
    if lat is None or lon is None:
        raise ValueError("위/경도 좌표를 찾을 수 없습니다 (latitude/longitude).")
    return lat, lon

def first_match(ds, candidates):
    lowers = {k.lower(): k for k in ds.data_vars}
    for nick in candidates:
        for var in ds.data_vars:
            if nick.lower() in var.lower():  # 부분 포함
                return var
        if nick.lower() in lowers:          # 정확 일치
            return lowers[nick.lower()]
    return None

def pick_main_o3(ds):
    v = first_match(ds, MAIN_O3_CANDS)
    if v: return v
    # fallback: "ozone" + "column"
    for var in ds.data_vars:
        lv = var.lower()
        if "ozone" in lv and "column" in lv:
            return var
    # fallback: "o3" + "column"
    for var in ds.data_vars:
        lv = var.lower()
        if "o3" in lv and "column" in lv:
            return var
    # fallback: "ozone"
    for var in ds.data_vars:
        if "ozone" in var.lower():
            return var
    return None

def infer_time(root, fname):
    s0 = root.attrs.get("time_coverage_start_since_epoch")
    s1 = root.attrs.get("time_coverage_end_since_epoch")
    if s0 is not None and s1 is not None:
        t0 = pd.to_datetime(float(s0), unit="s", utc=True)
        t1 = pd.to_datetime(float(s1), unit="s", utc=True)
        tm = t0 + (t1 - t0)/2
        return t0, t1, tm
    for cand in ["time", "Time", "scan_time"]:
        if cand in root.variables or cand in root.coords:
            try:
                vals = pd.to_datetime(root[cand].values, utc=True)
                tm = vals if vals.ndim == 0 else vals[0]
                tm = pd.to_datetime(tm, utc=True)
                return tm, tm, tm
            except Exception:
                pass
    m = re.search(r"_(\d{8}T\d{6})Z", fname)
    if m:
        tm = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S", utc=True)
        return tm, tm, tm
    tm = pd.Timestamp.utcnow().tz_localize("UTC")
    return tm, tm, tm

def ensure_coords(da, latname, lonname, root, prod):
    if latname in da.coords and lonname in da.coords:
        return da
    # root 우선 → prod 보조
    lat = (root[latname] if (latname in root.variables or latname in root.coords)
           else prod[latname])
    lon = (root[lonname] if (lonname in root.variables or lonname in root.coords)
           else prod[lonname])
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

# ===== 메인 =====
def main():
    files = [f for f in sorted(os.listdir(IN_DIR)) if f.lower().endswith(".nc")]
    if not files:
        print("입력 폴더에 .nc 파일이 없습니다.")
        return

    all_rows = []

    for fname in files:
        path = os.path.join(IN_DIR, fname)
        print(f"\n[처리] {fname}")
        try:
            root, prod = open_root_and_product(path)
            # 핵심 포인트: lat/lon은 root에서라도 반드시 찾아서 사용
            latname, lonname = find_lat_lon_any(root, prod)

            main_var = pick_main_o3(prod)
            if main_var is None:
                print(" - 총오존 변수 탐지 실패 → 건너뜀")
                root.close()
                if prod is not root: prod.close()
                continue

            t0, t1, tm = infer_time(root, fname)

            da_main = prod[main_var]
            da_main = ensure_coords(da_main, latname, lonname, root, prod)
            da_main = apply_bbox(da_main, latname, lonname, BBOX)

            df = da_main.to_dataframe(name="total_ozone_column").reset_index()
            if "time" not in df.columns:
                df["time"] = tm
            df = df.dropna(subset=["total_ozone_column"])

            # 보조 변수(있을 때만)
            df = add_optional(prod, df, "total_ozone_column_precision", PRECISION_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "effective_cloud_fraction", ECF_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "radiative_cloud_fraction", RCF_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "cloud_optical_centroid_pressure", OCP_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "solar_zenith_angle", SZA_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "viewing_zenith_angle", VZA_CANDS, latname, lonname, root)
            df = add_optional(prod, df, "qa_value", QA_CANDS, latname, lonname, root)

            # 메타
            df["time_start_utc"] = t0.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            df["time_end_utc"]   = t1.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            df["time_mid_utc"]   = tm.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            df["source_file"]    = fname
            df["units"]          = da_main.attrs.get("units", "")  # 보통 DU
            df["product_kind"]   = "o3"

            # 컬럼 순서 (있는 것만)
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
            tail = ["time_start_utc","time_end_utc","time_mid_utc","source_file","units","product_kind"]
            df = df[[c for c in base + extras + tail if c in df.columns]]

            all_rows.append(df)

            root.close()
            if prod is not root: prod.close()

        except Exception as e:
            print(f" - 오류: {e}")
            continue

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        out_path = os.path.join(OUT_DIR, OUT_CSV)
        out.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n완료: {len(out):,}개 행 → {out_path}")
    else:
        print("병합할 데이터가 없습니다.")

if __name__ == "__main__":
    main()
