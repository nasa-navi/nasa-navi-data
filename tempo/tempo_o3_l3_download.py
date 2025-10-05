import os
import earthaccess

# 1) Earthdata 로그인
ok = earthaccess.login(persist=True)
if not ok:
    raise RuntimeError("Earthdata 로그인 실패")

# 2) TEMPO NO2 L3 V03 설정
# 2) TEMPO NO2 L3 V03 설정
CONCEPT_ID = "C2930764281-LARC_CLOUD"   # TEMPO_NO2_L3_V03
OUTROOT = r""  # 경로 수정
BBOX = (-74.3, 40.4, -73.6, 41.0)       # NYC
START_DATE = "2025-06-01"
END_DATE   = "2025-06-10"

os.makedirs(OUTROOT, exist_ok=True)
print(f"\n=== TEMPO NO₂ L3 V03 검색: {START_DATE} ~ {END_DATE}, BBOX={BBOX} ===")

# 3) 데이터 검색
results = earthaccess.search_data(
    concept_id=CONCEPT_ID,
    temporal=(START_DATE, END_DATE),
    bounding_box=BBOX
)
print(f"▶ 발견된 granule 수: {len(results)}")

if not results:
    print(" 해당 기간/영역에 데이터가 없습니다.")
else:
    # 이미 받은 파일 제외
    have = set(os.listdir(OUTROOT))
    filtered = []
    for g in results:
        links = g.data_links()
        if not links:
            continue
        fname = links[0].rsplit("/", 1)[-1]
        if fname not in have:
            filtered.append(g)
    print(f"▶ 다운로드 대상: {len(filtered)} (이미 존재 {len(results)-len(filtered)}개 제외)")

    # 4) 다운로드 실행
    files = earthaccess.download(
        filtered,
        local_path=OUTROOT,  
        threads=8            # 병렬 다운로드 
    )

    # 5) 결과 리포트
    got = [f for f in files if f]
    print(f"\n 다운로드 완료: {len(got)}개 파일 저장 완료")
    print(f" 저장 폴더: {OUTROOT}")
