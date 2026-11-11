"""
기후 분석·예측 AI
- 1~6단계: 실제 기온 데이터로 파이썬이 직접 계산 (Open-Meteo 무료 API, 키 필요 없음)
- 7~8단계: 계산 결과를 Gemini API(구글 AI)에 보내서 해설과 대화
- 대화 중에 새 지역·연도·계절을 물으면 2~6단계로 돌아가 다시 계산 (순서도의 '새 지역·연도 질문?')
- 연평균뿐 아니라 봄·여름·가을·겨울 평균기온도 분석할 수 있다.

준비물
  py -m pip install google-genai requests matplotlib
  API 키는 코드에 쓰지 말고 환경변수 GEMINI_API_KEY 에 넣어 두세요.
  (키 만들기: https://aistudio.google.com -> Get API key, 무료 사용량 있음)
"""

import os
import sys
import json
import time
import requests
import matplotlib.pyplot as plt
from google import genai
from google.genai import types, errors

# ---------------- 설정 ----------------
START_YEAR = 1980                 # 과거 데이터 시작 연도
END_YEAR = 2025                   # 과거 데이터 마지막 연도
MODEL = "gemini-3.5-flash-lite"   # 무료 사용량이 있는 가장 저렴한 모델

# 계절 = 달 묶음. 겨울은 12월 + 다음 해 1·2월 (예: 2020년 겨울 = 2019년 12월 ~ 2020년 2월)
SEASONS = {"봄": (3, 4, 5), "여름": (6, 7, 8), "가을": (9, 10, 11), "겨울": (12, 1, 2)}
PERIODS = ["연평균", "봄", "여름", "가을", "겨울"]

# 날씨 사이트는 한국어 지역 이름을 잘 못 찾아서, 영어 이름으로 바꿔서 검색한다.
CITY_NAMES = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu", "인천": "Incheon",
    "대전": "Daejeon", "울산": "Ulsan", "제주": "Jeju", "수원": "Suwon",
    "춘천": "Chuncheon", "강릉": "Gangneung", "청주": "Cheongju", "전주": "Jeonju",
    "포항": "Pohang", "창원": "Changwon", "여수": "Yeosu", "목포": "Mokpo",
    "광주": "Gwangju", "천안": "Cheonan",
}

# 자주 쓰는 도시의 연도별 평균기온을 미리 저장해 둔 파일 (make_data.py 로 만든다).
# 날씨 사이트는 1분에 받을 수 있는 양이 정해져 있어서, 미리 받아 두면 빠르고 안정적이다.
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yearly_temps.json")


def load_saved_data():
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


SAVED = load_saved_data()


def period_label(period):
    """'연평균' -> '연평균기온', '여름' -> '여름 평균기온'"""
    return "연평균기온" if period == "연평균" else f"{period} 평균기온"


def find_saved(city, period):
    """저장된 도시면 (지역 이름, 연도 목록, 평균기온 목록)을, 아니면 None 을 돌려준다.
    '서울', '서울특별시', 'Seoul' 모두 찾는다."""
    key = city.strip().replace("특별시", "").replace("광역시", "").rstrip("시")
    for name, d in SAVED.items():
        if key == name or key.lower() == CITY_NAMES.get(name, "").lower():
            if period in d:
                return name, d[period]["years"], d[period]["avgs"]
    return None


# ---------------- 1단계: 입력 ----------------
def get_input():
    city = input("분석할 지역을 입력하세요 (예: 서울): ").strip()
    while True:
        try:
            year = int(input(f"예측할 미래 연도를 입력하세요 ({END_YEAR + 1} 이후): "))
            if year > END_YEAR:
                break
            print(f"{END_YEAR}년보다 뒤의 연도를 입력해 주세요.")
        except ValueError:
            print("숫자로 입력해 주세요.")
    while True:
        period = input("계절을 입력하세요 (연평균/봄/여름/가을/겨울, 그냥 Enter = 연평균): ").strip()
        if period == "":
            period = "연평균"
        if period in PERIODS:
            return city, year, period
        print("연평균, 봄, 여름, 가을, 겨울 중에서 골라 주세요.")


# ---------------- 2단계: 데이터 수집 ----------------
def find_location(city):
    """지역 이름 -> 위도, 경도"""
    key = city.replace("특별시", "").replace("광역시", "").rstrip("시")
    search_name = CITY_NAMES.get(key, city)   # 표에 없으면 입력한 그대로 검색
    url = "https://geocoding-api.open-meteo.com/v1/search"
    res = requests.get(url, params={"name": search_name, "count": 1, "language": "ko"}, timeout=20)
    results = res.json().get("results")
    if not results:
        return None
    r = results[0]
    return r["latitude"], r["longitude"], r.get("name", city)


def load_temperature(lat, lon):
    """과거 일별 평균기온 데이터 불러오기"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{START_YEAR}-01-01",
        "end_date": f"{END_YEAR}-12-31",
        "daily": "temperature_2m_mean",
        "timezone": "Asia/Seoul",
    }
    for wait in (0, 20, 40):           # 너무 자주 요청해서 거절(429)되면 기다렸다가 다시
        time.sleep(wait)
        res = requests.get(url, params=params, timeout=60)
        if res.status_code != 429:
            break
    data = res.json()
    if "daily" not in data:
        return None
    return data["daily"]["time"], data["daily"]["temperature_2m_mean"]


# ---------------- 3단계: 데이터 전처리 ----------------
def make_table(dates, temps, period="연평균"):
    """빠진 값은 빼고, 연도별 평균기온 표로 정리. period 가 계절이면 그 계절의 달만 쓴다."""
    months = SEASONS.get(period)      # 연평균이면 None (모든 달 사용)
    by_year = {}
    for d, t in zip(dates, temps):
        if t is None:                 # 누락값 제거
            continue
        if t < -60 or t > 60:         # 비정상 값 제거
            continue
        year, month = int(d[:4]), int(d[5:7])
        if months is not None:
            if month not in months:
                continue
            if month == 12:           # 12월은 다음 해 겨울로 센다
                year += 1
        by_year.setdefault(year, []).append(t)

    need = 300 if months is None else 80   # 데이터가 너무 적은 해는 제외 (1년 365일, 한 계절 약 90일)
    years, avgs = [], []
    for y in sorted(by_year):
        if y > END_YEAR or len(by_year[y]) < need:
            continue
        years.append(y)
        avgs.append(sum(by_year[y]) / len(by_year[y]))
    return years, avgs


# ---------------- 4단계: 추세 분석 (선형회귀) ----------------
def linear_regression(xs, ys):
    """점들 가운데를 지나는 직선 y = a*x + b 구하기 (최소제곱법)"""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    top = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    bottom = sum((x - mean_x) ** 2 for x in xs)
    a = top / bottom          # 기울기: 1년에 몇 도 오르는지
    b = mean_y - a * mean_x
    return a, b


# ---------------- 5단계: 미래 예측 ----------------
def predict(a, b, year):
    return a * year + b


# ---------------- 6단계: 위험도 판단 ----------------
def judge_risk(rise):
    if rise < 1:
        return "주의"
    elif rise < 2:
        return "경고"
    else:
        return "위험"


# ---------------- 그래프 (콘솔 버전) ----------------
def show_graph(r):
    label = period_label(r["period"])
    plt.rcParams["font.family"] = "Malgun Gothic"   # 윈도우 한글 글꼴
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(9, 5))
    plt.plot(r["years"], r["avgs"], "o-", label=f"실제 {label}")
    line_x = [r["years"][0], r["target_year"]]
    plt.plot(line_x, [predict(r["a"], r["b"], x) for x in line_x], "r--", label="추세선(예측)")
    plt.plot(r["target_year"], r["future_temp"], "r*", markersize=15, label=f"{r['target_year']}년 예측")
    plt.title(f"{r['place']} {label} 변화와 예측")
    plt.xlabel("연도")
    plt.ylabel("기온(℃)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


# ---------------- 2~6단계를 한 번에 실행 (시뮬레이션) ----------------
def simulate(city, target_year, period="연평균"):
    """2~6단계 계산만 한다. 콘솔 버전과 웹 버전이 같이 쓴다.
    성공하면 (결과, None), 실패하면 (None, 오류 메시지)를 돌려준다."""
    if target_year <= END_YEAR:
        return None, f"{END_YEAR}년보다 뒤의 연도만 예측할 수 있어요."
    if period not in PERIODS:
        period = "연평균"

    saved = find_saved(city, period)                      # 2~3단계: 저장된 도시면 파일에서
    if saved:
        place, years, avgs = saved
    else:                                                 #          아니면 인터넷에서
        try:
            loc = find_location(city)                     # 2단계
            if loc is None:
                return None, f"'{city}' 지역을 찾지 못했습니다. 영어 이름(예: Seoul)으로 해 보세요."
            lat, lon, place = loc
            data = load_temperature(lat, lon)
        except requests.RequestException:
            return None, "인터넷에서 데이터를 불러오지 못했습니다. 잠시 후 다시 해 보세요."
        if data is None:
            return None, "기온 데이터를 받지 못했습니다. 잠시 후 다시 해 보세요."
        years, avgs = make_table(*data, period)           # 3단계

    a, b = linear_regression(years, avgs)                 # 4단계
    past_change = a * (years[-1] - years[0])              #   과거에 추세상 몇 도 변했는지

    now_temp = predict(a, b, years[-1])                   # 5단계
    future_temp = predict(a, b, target_year)
    rise = future_temp - now_temp

    risk = judge_risk(rise)                               # 6단계

    return {
        "place": place, "period": period, "first_year": years[0], "last_year": years[-1],
        "past_change": past_change, "per_decade": a * 10, "now_temp": now_temp,
        "target_year": target_year, "future_temp": future_temp, "rise": rise, "risk": risk,
        "years": years, "avgs": avgs, "a": a, "b": b,      # 그래프용
    }, None


def run_simulation(city, target_year, period="연평균"):
    """콘솔 버전: 계산하고 결과를 화면에 출력 + 그래프. 실패하면 None."""
    print(f"\n{city}, {target_year}년, {period} 시뮬레이션: 데이터를 불러오는 중...")
    r, error = simulate(city, target_year, period)
    if error:
        print(error)
        return None

    print(f"\n=== {r['place']} {period_label(r['period'])} 분석 결과 ===")
    print(f"과거 변화({r['first_year']}~{r['last_year']}년): {r['past_change']:+.2f}℃")
    print(f"10년당 상승폭: {r['per_decade']:+.2f}℃")
    print(f"{r['target_year']}년 예상 기온: {r['future_temp']:.2f}℃ (현재 대비 {r['rise']:+.2f}℃)")
    print(f"위험도: {r['risk']}")

    show_graph(r)
    return r


# ---------------- 전국 지도용: 여러 도시를 같은 방법으로 계산 ----------------
# 도시 이름: (위도, 경도). 지도 위 위치를 정확히 하려고 좌표를 직접 적어 둔다.
MAP_CITIES = {
    "서울": (37.57, 126.98), "인천": (37.46, 126.71), "수원": (37.26, 127.03),
    "춘천": (37.88, 127.73), "강릉": (37.75, 128.88), "청주": (36.64, 127.49),
    "천안": (36.82, 127.11), "대전": (36.35, 127.38), "전주": (35.82, 127.15),
    "광주": (35.16, 126.85), "목포": (34.81, 126.39), "여수": (34.76, 127.66),
    "대구": (35.87, 128.60), "포항": (36.02, 129.34), "울산": (35.54, 129.31),
    "부산": (35.18, 129.08), "창원": (35.23, 128.68), "제주": (33.50, 126.53),
}


def trend_of(years, avgs):
    """연도별 평균기온으로 기온 추세 계산 (4단계)"""
    a, b = linear_regression(years, avgs)
    return {
        "per_decade": a * 10,                             # 10년당 상승폭
        "past_change": a * (years[-1] - years[0]),        # 1980~2025 추세상 변화
        "avg_temp": sum(avgs) / len(avgs),                # 기간 평균기온
    }


def result_text(r):
    """계산 결과를 AI에게 보낼 글로 바꾸기"""
    return f"""[분석 결과]
- 지역: {r['place']}
- 분석한 기온: {period_label(r['period'])}
- 분석 기간: {r['first_year']}~{r['last_year']}년
- 과거 변화: 분석 기간 동안 추세상 {r['past_change']:+.2f}℃
- 10년당 기온 상승폭: {r['per_decade']:+.2f}℃
- {r['last_year']}년 추세 기온: {r['now_temp']:.2f}℃
- {r['target_year']}년 예상 기온: {r['future_temp']:.2f}℃
- 예상 상승폭: {r['rise']:+.2f}℃
- 위험도: {r['risk']}"""


# ---------------- 7단계: AI 해설 생성 ----------------
SYSTEM_PROMPT = """너는 기후 분석 결과를 설명해 주는 'AI 기후 해설가'야.
대화 중에 [분석 결과]가 주어져. 그 숫자는 실제 관측 데이터로 파이썬이 계산한 값이야.
분석한 기온은 연평균기온일 수도 있고, 봄·여름·가을·겨울 중 한 계절의 평균기온일 수도 있어.

[숫자 규칙]
- 기온, 상승폭 같은 숫자는 [분석 결과]에 있는 것만 써. 바꾸거나 새로 지어내지 마.
- [분석 결과]에 없는 숫자를 물으면 모른다고 말하고,
  '2080년 부산은?', '여름은?'처럼 지역·연도·계절을 넣어서 물어보면 새로 계산할 수 있다고 안내해.

[설명 규칙]
- '왜?', '어떻게?' 같은 질문에는 일반적으로 알려진 과학 원리로 자신 있게 설명해 줘.
  (예: 바다는 비열이 커서 천천히 데워진다, 도시는 열섬 현상이 있다, 온실가스가 열을 가둔다)
- 계절 결과라면 그 계절에 맞는 영향과 예방 방법을 말해 줘 (예: 여름 폭염, 겨울 한파·적설 감소).
- 단, 원리 설명에도 새 숫자는 넣지 마.
- 고등학생이 이해할 수 있게 쉽고 짧게 설명해.
위험도 기준: 상승폭 1℃ 미만 주의, 1~2℃ 경고, 2℃ 이상 위험.
예측은 단순 선형 추세로 계산한 값이라 실제 미래와 다를 수 있어."""


def explain(chat, message):
    """AI 해설 받기. 실패하면 이유를 알려 주는 문장을 돌려준다."""
    try:
        response = chat.send_message(message)   # chat 이 이전 대화를 기억하고 있음
        return response.text
    except errors.APIError as e:
        return f"(AI 해설을 받지 못했습니다. API 키, 인터넷, 무료 사용량을 확인해 주세요. 오류: {e})"


# ---------------- 8단계 마름모: 새 지역·연도 질문? ----------------
# AI에게 '시뮬레이션 실행' 도구를 알려 주고, AI가 그 도구를 쓰겠다고 하면 '예'로 본다.
# AI는 지역·연도·계절만 골라 주고, 실제 계산은 simulate() 가 한다.
SIMULATION_TOOL = types.Tool(function_declarations=[{
    "name": "run_simulation",
    "description": "사용자가 다른 지역, 다른 미래 연도, 다른 계절의 기온 예측을 원할 때 쓴다.",
    "parameters_json_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "지역 이름 (영어로, 예: Seoul, Busan)"},
            "year": {"type": "integer", "description": "예측할 미래 연도"},
            "period": {"type": "string", "enum": PERIODS,
                       "description": "분석할 기온: 연평균, 봄, 여름, 가을, 겨울 중 하나"},
        },
        "required": ["city", "year", "period"],
    },
}])


def check_new_request(client, question, result):
    """새 지역·연도·계절 질문이면 (지역, 연도, 계절)을, 아니면 None 을 돌려준다."""
    rule = f"""너는 사용자 질문을 분류하는 역할이야.
지금 분석한 것은 '{result['place']}'의 {result['target_year']}년 {period_label(result['period'])} 예측이야.
사용자가 지금과 다른 지역, 다른 연도, 다른 계절(연평균/봄/여름/가을/겨울)의 기온 예측을 원할 때만
run_simulation 도구를 써. 지역·연도·계절 중 말하지 않은 것은 지금 값을 그대로 써.
지역·연도·계절이 지금과 모두 똑같으면 도구를 쓰지 마.
최고기온, 강수량, 이유 같은 다른 질문이면 도구를 쓰지 말고 '아니오'라고만 답해."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=question,
            config=types.GenerateContentConfig(system_instruction=rule, tools=[SIMULATION_TOOL]),
        )
    except errors.APIError:
        return None                       # 판단에 실패하면 '아니오'로 처리
    if response.function_calls:           # AI가 도구를 쓰겠다고 함 -> '예'
        args = response.function_calls[0].args
        return args["city"], int(args["year"]), args.get("period", result["period"])
    return None


# ---------------- 전체 실행 (순서도 그대로) ----------------
def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY 환경변수가 없습니다. 먼저 API 키를 설정해 주세요.")
        sys.exit()
    client = genai.Client()  # 환경변수 GEMINI_API_KEY 를 자동으로 읽음
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )

    city, target_year, period = get_input()               # 1단계
    result = run_simulation(city, target_year, period)    # 2~6단계
    if result is None:
        return

    message = (result_text(result)
               + "\n\n이 결과로 미래 기후가 어떻게 될지, 그리고 우리가 할 수 있는 예방 방법을 설명해 줘.")

    while True:
        answer = explain(chat, message)                   # 7단계: AI 해설 생성
        print("\n[AI 기후 해설가]\n" + answer + "\n")      #        출력

        question = input("추가 질문 ('종료' 입력 시 끝): ").strip()
        if question == "종료":                             # '종료' 입력?
            print("프로그램을 종료합니다.")
            break

        new = check_new_request(client, question, result)  # 새 지역·연도 질문?
        if new is not None:                                 # 예 -> 2~6단계 다시
            new_result = run_simulation(*new)
            if new_result is not None:
                result = new_result
                question = question + "\n\n" + result_text(result)
        message = question                                  # 아니오 -> 7단계로


if __name__ == "__main__":
    main()
