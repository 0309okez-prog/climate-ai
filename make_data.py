"""
전국 지도용 도시들의 연도별 평균기온(연평균 + 봄·여름·가을·겨울)을 미리 받아서
yearly_temps.json 에 저장한다. (순서도 2~3단계를 미리 해 두는 것)
1980~2025년 과거 데이터라 한 번만 받으면 된다.

실행:  py make_data.py
날씨 사이트가 1분에 받을 수 있는 양이 정해져 있어서, 도시마다 잠깐씩 쉬면서 받는다. (5분쯤)
"""

import json
import time
from climate_ai import (MAP_CITIES, PERIODS, DATA_FILE, load_temperature, make_table,
                        load_saved_data)

saved = load_saved_data()
for name, (lat, lon) in MAP_CITIES.items():
    if name in saved and all(p in saved[name] for p in PERIODS):
        print(f"{name}: 이미 있음")                         # 이미 받은 도시는 건너뜀
        continue
    print(f"{name}: 받는 중...", end=" ", flush=True)
    data = None
    for attempt in range(3):
        try:
            data = load_temperature(lat, lon)               # 2단계
        except Exception as e:
            print(f"(오류: {e})", end=" ")
        if data:
            break
        time.sleep(60)                                      # 거절되면 1분 쉬고 다시
    if not data:
        print("실패 - 나중에 다시 실행하세요")
        continue
    city = {"lat": lat, "lon": lon}
    for period in PERIODS:                                  # 3단계: 연평균 + 계절별 표
        years, avgs = make_table(*data, period)
        city[period] = {"years": years, "avgs": [round(t, 3) for t in avgs]}
    saved[name] = city
    with open(DATA_FILE, "w", encoding="utf-8") as f:       # 한 도시 받을 때마다 저장
        json.dump(saved, f, ensure_ascii=False)
    print(f"완료 ({len(city['연평균']['years'])}년치)")
    time.sleep(15)                                          # 다음 도시 전에 잠깐 쉬기

print(f"\n저장 완료: {len(saved)}/{len(MAP_CITIES)}개 도시 -> {DATA_FILE}")
