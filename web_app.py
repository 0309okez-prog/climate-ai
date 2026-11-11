"""
기후 분석·예측 AI - 웹 버전 (Streamlit)
- 계산(2~6단계)과 AI(7~8단계) 함수는 climate_ai.py 것을 그대로 쓴다.
- 이 파일은 '화면'만 담당한다. 순서도는 콘솔 버전과 같다.
- 화면 구성: 위쪽 소개 + 카드 3개 (지역 분석·AI 대화 / 전국 지도 / 계절 비교)

내 컴퓨터에서 실행:  py -m streamlit run web_app.py
"""

import os
import re
import pandas as pd
import altair as alt
import pydeck as pdk
import streamlit as st
from google import genai
from google.genai import types, errors

from climate_ai import (END_YEAR, MODEL, SYSTEM_PROMPT, PERIODS, SEASONS, SAVED, MAP_CITIES,
                        simulate, predict, trend_of, period_label,
                        result_text, explain, check_new_request)

st.set_page_config(page_title="기후 분석·예측 AI", page_icon="🌏", layout="wide")

RISK_ICON = {"주의": "🟢", "경고": "🟠", "위험": "🔴"}
PERIOD_COLOR = {"연평균": "#4c9be8", "봄": "#6fcf7f", "여름": "#ff6b6b", "가을": "#f4a259", "겨울": "#7cc4ff"}
FIRST_REQUEST = "이 결과로 미래 기후가 어떻게 될지, 그리고 우리가 할 수 있는 예방 방법을 설명해 줘."
PAGES = [  # (이름, 아이콘, 제목, 설명)
    ("analysis", "🔍", "지역 분석·AI 대화", "한 지역의 과거와 미래 기온을 계산하고 AI와 대화해요"),
    ("map", "🗺️", "전국 지도", f"전국 {len(MAP_CITIES)}개 도시의 기온 상승을 한눈에 비교해요"),
    ("season", "🍂", "계절 비교", "봄·여름·가을·겨울 중 어느 계절이 더 빨리 따뜻해질까요?"),
]

# ---------------- 화면 꾸미기 (CSS) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea { font-family: 'Noto Sans KR', sans-serif; }
.stApp {   /* 우주에서 본 지구처럼: 아래쪽에 푸른 빛, 위쪽은 어두운 남색 */
  background:
    radial-gradient(ellipse 110% 55% at 50% 118%, rgba(45,130,255,.55), rgba(20,60,140,.15) 55%, transparent 70%),
    radial-gradient(ellipse 60% 40% at 90% -5%, rgba(110,80,220,.25), transparent 70%),
    radial-gradient(1px 1px at 12% 18%, #fff8, transparent), radial-gradient(1px 1px at 72% 12%, #fff6, transparent),
    radial-gradient(1px 1px at 40% 30%, #fff5, transparent), radial-gradient(1px 1px at 85% 40%, #fff6, transparent),
    #060d1c;
  background-attachment: fixed;
}
.hero-badge { display:inline-block; padding:.25rem .8rem; border:1px solid #ffffff33; border-radius:999px;
              font-size:.8rem; color:#b9c8e4; background:#ffffff0d; }
.hero h1 { font-size:3.1rem; font-weight:900; line-height:1.15; margin:.8rem 0 1rem; color:#fff;
           text-shadow:0 4px 24px rgba(60,140,255,.45); }
.hero p { color:#c5d3ea; font-size:1.05rem; line-height:1.75; }
[class*="st-key-card_"] { border-radius:18px; padding:1.2rem 1rem 1rem; min-height:250px;
    border:2px solid #ffffff14; box-shadow:0 12px 32px rgba(0,0,0,.45); transition:transform .15s; }
[class*="st-key-card_"]:hover { transform:translateY(-5px); }
[class*="st-key-card_analysis"] { background:linear-gradient(165deg,#2a6fd0 0%,#123a73 55%,#0b2142 100%); }
[class*="st-key-card_map"]      { background:linear-gradient(165deg,#2f9b6f 0%,#155443 55%,#0b2a24 100%); }
[class*="st-key-card_season"]   { background:linear-gradient(165deg,#c0702f 0%,#6b3b5c 55%,#2a1838 100%); }
[class*="st-key-card_"][class*="_on"] { border-color:#9fd8ff; box-shadow:0 0 0 3px #9fd8ff55, 0 12px 32px rgba(0,0,0,.45); }
.card-icon { font-size:3rem; line-height:1; margin-bottom:.8rem; }
.card-title { font-size:1.12rem; font-weight:700; color:#fff; margin-bottom:.35rem; }
.card-desc { font-size:.85rem; color:#e3ebf8; opacity:.85; min-height:3.2em; }
.section-title { font-size:1.6rem; font-weight:800; margin:.3rem 0 .2rem; }
</style>
""", unsafe_allow_html=True)


# ---------------- 도움 함수 ----------------
def get_secret(name):
    """웹 서버 설정(Secrets)이나 환경변수에서 값 읽기. 키는 코드에 쓰지 않는다."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:          # 내 컴퓨터처럼 secrets 설정 파일이 없을 때
        pass
    return os.environ.get(name)


def show_text(text):
    """AI 글 보여 주기. '**+0.29℃**씩'처럼 굵은 글씨 바로 뒤에 한글이 붙으면
    별표가 그대로 보이는 문제가 있어서, 굵은 글씨를 HTML <b>로 바꿔서 보여 준다."""
    text = text.replace("<", "&lt;")                     # AI 글 속 HTML은 글자로만 보이게
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    st.markdown(text, unsafe_allow_html=True)


def ask_ai_once(client, request):
    """대화 기록 없이 AI에게 한 번 물어보기 (지도·계절 해설용)"""
    try:
        response = client.models.generate_content(
            model=MODEL, contents=request,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT))
        return response.text
    except errors.APIError as e:
        return f"(AI 해설을 받지 못했습니다. 오류: {e})"


@st.cache_data(show_spinner=False)
def cached_simulate(city, year, period):
    """같은 지역·연도·계절은 다시 계산하지 않고 저장해 둔 결과를 쓴다."""
    result, error = simulate(city, year, period)
    if error:
        raise ValueError(error)  # 실패한 결과는 저장하지 않도록 오류로 알림
    return result


def draw_chart(r, height=320):
    """실제 기온 + 추세선 + 예측 지점 그래프"""
    years = list(range(r["first_year"], r["target_year"] + 1))
    actual = dict(zip(r["years"], r["avgs"]))
    label = f"실제 {period_label(r['period'])}"
    rows = [{"year": y, "temp": actual[y], "kind": label} for y in years if y in actual]
    rows += [{"year": y, "temp": predict(r["a"], r["b"], y), "kind": "추세선(예측)"} for y in years]
    lines = alt.Chart(pd.DataFrame(rows)).mark_line().encode(
        x=alt.X("year:Q", title="연도", axis=alt.Axis(format="d")),
        y=alt.Y("temp:Q", title="기온(℃)", scale=alt.Scale(zero=False)),  # 0℃부터 시작하지 않게
        color=alt.Color("kind:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=[label, "추세선(예측)"],
                                        range=[PERIOD_COLOR[r["period"]], "#ff5a5a"])),
        strokeDash=alt.StrokeDash("kind:N", legend=None,
                                  scale=alt.Scale(domain=[label, "추세선(예측)"], range=[[1, 0], [6, 4]])),
    )
    star = alt.Chart(pd.DataFrame([{"year": r["target_year"], "temp": r["future_temp"]}])).mark_point(
        shape="diamond", size=200, filled=True, color="#ff5a5a",
    ).encode(x="year:Q", y="temp:Q", tooltip=[alt.Tooltip("temp:Q", title="예측 기온", format=".2f")])
    st.altair_chart((lines + star).properties(height=height), use_container_width=True)


# ---------------- 페이지 1: 지역 분석·AI 대화 (순서도 1~8단계) ----------------
def page_analysis(client):
    st.markdown('<div class="section-title">🔍 지역 분석·AI 대화</div>', unsafe_allow_html=True)

    with st.form("input"):                                   # 1단계: 입력
        c1, c2, c3 = st.columns([1, 1, 2])
        city = c1.text_input("지역", value="서울")
        year = c2.number_input("예측할 미래 연도", min_value=END_YEAR + 1, max_value=2200,
                               value=2050, step=1)
        period = c3.radio("기온 종류", PERIODS, horizontal=True)
        submitted = st.form_submit_button("분석 시작", type="primary")

    if submitted:
        with st.spinner("데이터를 불러와 계산하는 중..."):
            try:
                result = cached_simulate(city.strip(), int(year), period)   # 2~6단계
            except ValueError as e:
                st.error(str(e))
                return
        st.session_state.result = result
        st.session_state.chat = client.chats.create(              # 새 대화 시작
            model=MODEL, config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT))
        with st.spinner("AI가 결과를 해설하는 중..."):
            answer = explain(st.session_state.chat,                # 7단계
                             result_text(result) + "\n\n" + FIRST_REQUEST)
        st.session_state.messages = [{"role": "assistant", "content": answer}]

    r = st.session_state.result
    if r is None:
        st.info("지역·연도·기온 종류를 고르고 **분석 시작**을 눌러 보세요.")
        return

    left, right = st.columns([1.15, 1], gap="large")
    with left:                                               # 결과: 숫자 + 그래프
        with st.container(border=True):
            st.subheader(f"📊 {r['place']} · {r['target_year']}년 · {period_label(r['period'])}")
            m1, m2, m3 = st.columns(3)
            m1.metric(f"과거 변화 ({r['first_year']}~{r['last_year']})", f"{r['past_change']:+.2f}℃")
            m2.metric(f"{r['target_year']}년 예상", f"{r['future_temp']:.2f}℃",
                      f"{r['rise']:+.2f}℃ (현재 대비)", delta_color="inverse")
            m3.metric("위험도", f"{RISK_ICON[r['risk']]} {r['risk']}")
            draw_chart(r)
            st.caption("※ 단순 선형 추세(최소제곱법)로 계산한 값이라 실제 미래와 다를 수 있어요. "
                       "위험도: 상승폭 1℃ 미만 주의 / 1~2℃ 경고 / 2℃ 이상 위험")

    with right:                                              # 7~8단계: AI 해설 + 대화
        with st.container(border=True):
            st.subheader("🤖 AI 기후 해설가")
            box = st.container(height=470, autoscroll=True)   # 새 답이 오면 아래로 자동 스크롤
            with box:
                for m in st.session_state.messages:
                    with st.chat_message(m["role"]):
                        show_text(m["content"])
            question = st.chat_input("예: 그럼 2080년 부산은? / 여름은? / 왜 더워져?")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        message = question
        with st.spinner("AI가 생각하는 중..."):
            new = check_new_request(client, question, r)          # 새 지역·연도·계절 질문?
            if new is not None:                                   # 예 -> 2~6단계 다시
                try:
                    new_result = cached_simulate(*new)
                    st.session_state.result = new_result
                    message = question + "\n\n" + result_text(new_result)
                    note = (f"📊 **{new_result['place']} {new_result['target_year']}년 "
                            f"{period_label(new_result['period'])}**으로 새로 계산했어요. 그래프가 바뀌었어요.")
                except ValueError as e:
                    note = f"⚠️ {e}"
                st.session_state.messages.append({"role": "assistant", "content": note})
            answer = explain(st.session_state.chat, message)      # 7단계
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()   # 바뀐 결과와 그래프로 화면 다시 그리기


# ---------------- 페이지 2: 전국 지도 ----------------
def page_map(client):
    st.markdown('<div class="section-title">🗺️ 전국 기온 상승 지도</div>', unsafe_allow_html=True)
    st.caption(f"전국 {len(SAVED)}개 도시의 1980~2025년 기온을 파이썬이 같은 방법(선형회귀)으로 계산해 비교해요.")
    if not SAVED:
        st.error("저장된 도시 데이터가 없어요. 먼저 `py make_data.py`를 실행해 주세요.")
        return

    c1, c2 = st.columns(2)
    period = c1.radio("기온 종류", PERIODS, horizontal=True, key="map_period")
    choice = c2.radio("무엇을 볼까요?", ["10년당 상승폭", "과거 변화 (1980~2025)"], horizontal=True)
    col = "per_decade" if choice == "10년당 상승폭" else "past_change"

    rows = []
    for name, d in SAVED.items():                            # 도시마다 4단계 계산
        t = trend_of(d[period]["years"], d[period]["avgs"])
        rows.append({"name": name, "lat": d["lat"], "lon": d["lon"], **t})
    df = pd.DataFrame(rows)

    low, high = df[col].min(), df[col].max()
    def color(v):                                            # 값이 클수록 진한 빨강
        t = (v - low) / (high - low) if high > low else 0.5
        return [255, int(210 - 170 * t), int(150 - 130 * t), 230]
    df["color"] = df[col].apply(color)
    df["label"] = df.apply(lambda row: f"{row['name']} {row[col]:+.2f}℃", axis=1)

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        circles = pdk.Layer("ScatterplotLayer", df, get_position=["lon", "lat"],
                            get_fill_color="color", get_radius=15000, pickable=True)
        labels = pdk.Layer("TextLayer", df, get_position=["lon", "lat"], get_text="label",
                           get_size=13, get_color=[255, 255, 255], get_pixel_offset=[0, -22],
                           # 지도에 쓸 글자 목록 (한글). 따옴표로 감싸야 pydeck이 글자 그대로 보낸다.
                           character_set='"' + "".join(sorted(set("".join(df["label"])))) + '"',
                           background=True, get_background_color=[10, 20, 40, 200])
        view = pdk.ViewState(latitude=35.9, longitude=127.8, zoom=5.8)
        st.pydeck_chart(pdk.Deck(layers=[circles, labels], initial_view_state=view, map_style="dark"))
    with right:
        ranked = df.sort_values(col, ascending=False)
        bars = alt.Chart(ranked).mark_bar(color=PERIOD_COLOR[period]).encode(
            x=alt.X(f"{col}:Q", title=f"{choice} (℃)"),
            y=alt.Y("name:N", title=None, sort=None, axis=alt.Axis(labelOverlap=False)),
            tooltip=["name", alt.Tooltip(f"{col}:Q", format="+.2f")],
        ).properties(height=480)
        st.altair_chart(bars, use_container_width=True)
        top, bottom = ranked.iloc[0], ranked.iloc[-1]
        st.markdown(f"**{period_label(period)}** 기준 가장 빨리 더워진 곳: **{top['name']} "
                    f"({top[col]:+.2f}℃)** · 가장 천천히: **{bottom['name']} ({bottom[col]:+.2f}℃)**")

    if st.button("🤖 AI에게 이 지도 설명 듣기"):              # 숫자는 파이썬 계산만 보냄
        table = "\n".join(f"- {row['name']}: 10년당 {row['per_decade']:+.2f}℃, "
                          f"1980~2025 변화 {row['past_change']:+.2f}℃" for _, row in ranked.iterrows())
        request = (f"[분석 결과: 전국 도시별 {period_label(period)} 추세]\n{table}\n\n"
                   "어떤 지역이 빨리 더워지고 있는지, 왜 그런지 과학 원리로 설명하고 예방 방법도 알려 줘.")
        with st.spinner("AI가 지도를 해설하는 중..."):
            st.session_state.map_answer = ask_ai_once(client, request)
    if st.session_state.get("map_answer"):
        with st.container(border=True):
            with st.chat_message("assistant"):
                show_text(st.session_state.map_answer)


# ---------------- 페이지 3: 계절 비교 ----------------
def page_season(client):
    st.markdown('<div class="section-title">🍂 계절 비교</div>', unsafe_allow_html=True)
    st.caption("같은 도시라도 계절마다 따뜻해지는 속도가 달라요. 파이썬이 계절별로 따로 계산해 비교해요. "
               "(겨울 = 12월 + 다음 해 1·2월)")
    if not SAVED:
        st.error("저장된 도시 데이터가 없어요. 먼저 `py make_data.py`를 실행해 주세요.")
        return

    city = st.selectbox("도시", list(SAVED.keys()))
    d = SAVED[city]
    seasons = list(SEASONS)
    trends = {p: trend_of(d[p]["years"], d[p]["avgs"]) for p in seasons}   # 계절마다 4단계

    m = st.columns(4)
    for col, p in zip(m, seasons):
        col.metric(f"{p} · 10년당", f"{trends[p]['per_decade']:+.2f}℃",
                   f"과거 변화 {trends[p]['past_change']:+.2f}℃", delta_color="inverse")

    grid = st.columns(2)
    for i, p in enumerate(seasons):                         # 계절별 작은 그래프 4개
        with grid[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{p}** 평균기온")
                r, _ = simulate(city, END_YEAR + 25, p)     # 2050년까지 추세선 연장
                draw_chart(r, height=220)

    fastest = max(seasons, key=lambda p: trends[p]["per_decade"])
    st.markdown(f"**{city}**에서 가장 빨리 따뜻해지는 계절은 **{fastest}** "
                f"(10년당 {trends[fastest]['per_decade']:+.2f}℃)예요.")

    if st.button("🤖 AI에게 계절 차이 설명 듣기"):
        table = "\n".join(f"- {p}: 10년당 {trends[p]['per_decade']:+.2f}℃, "
                          f"1980~2025 변화 {trends[p]['past_change']:+.2f}℃" for p in seasons)
        request = (f"[분석 결과: {city}의 계절별 평균기온 추세]\n{table}\n\n"
                   "계절마다 따뜻해지는 속도가 왜 다른지 과학 원리로 설명하고, 계절별 예방 방법도 알려 줘.")
        with st.spinner("AI가 계절 차이를 해설하는 중..."):
            st.session_state.season_answer = (city, ask_ai_once(client, request))
    saved_answer = st.session_state.get("season_answer")
    if saved_answer and saved_answer[0] == city:
        with st.container(border=True):
            with st.chat_message("assistant"):
                show_text(saved_answer[1])


# ---------------- 맨 위: 소개 + 카드 ----------------
if "page" not in st.session_state:
    st.session_state.page = "analysis"
    st.session_state.result = None    # 지금 보고 있는 분석 결과
    st.session_state.messages = []    # 화면에 보여 줄 대화

hero, cards = st.columns([1, 1.55], gap="large")
with hero:
    st.markdown("""
<div class="hero">
  <span class="hero-badge">🌏 기후분석과 예방 · 소프트웨어 프로젝트</span>
  <h1>기후 분석·예측 AI</h1>
  <p>우리 지역의 <b>1980년부터 2025년까지</b> 실제 기온을 분석하고,<br>
  미래 기온을 예측해요. 숫자는 <b>파이썬이 계산</b>하고,<br>
  <b>AI가 쉽게 설명</b>하고 예방 방법을 알려 줘요.</p>
</div>""", unsafe_allow_html=True)

with cards:
    for col, (pid, icon, title, desc) in zip(st.columns(3), PAGES):
        with col:
            on = "_on" if st.session_state.page == pid else ""
            with st.container(key=f"card_{pid}{on}"):
                st.markdown(f'<div class="card-icon">{icon}</div><div class="card-title">{title}</div>'
                            f'<div class="card-desc">{desc}</div>', unsafe_allow_html=True)
                if st.button("열기 →", key=f"open_{pid}", use_container_width=True):
                    st.session_state.page = pid
                    st.rerun()

st.divider()

# ---------------- 비밀번호 (모르는 사람이 무료 사용량을 쓰지 못하게) ----------------
password = get_secret("APP_PASSWORD")
if password and not st.session_state.get("unlocked"):
    entered = st.text_input("비밀번호를 입력하세요", type="password")
    if entered == password:
        st.session_state.unlocked = True
        st.rerun()
    elif entered:
        st.error("비밀번호가 틀렸어요.")
    st.stop()

api_key = get_secret("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY가 설정되지 않았습니다. 관리자에게 알려 주세요.")
    st.stop()
if "client" not in st.session_state:
    st.session_state.client = genai.Client(api_key=api_key)
client = st.session_state.client

if st.session_state.page == "analysis":
    page_analysis(client)
elif st.session_state.page == "map":
    page_map(client)
else:
    page_season(client)

st.caption("데이터: Open-Meteo (과거 기온 재분석 자료) · AI: Google Gemini · "
           "숫자는 모두 파이썬 계산, AI는 설명만 해요.")
