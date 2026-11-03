import base64
import os

import joblib
import altair as alt
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="Bike Rental Demand Predictor", page_icon="🚲", layout="wide")


# ------------------------------------------------------------------ model
@st.cache_resource
def load_model():
    # Native XGBoost JSON works across library versions; .pkl is the fallback
    if os.path.exists("champion_bike_rental_model.json"):
        m = XGBRegressor()
        m.load_model("champion_bike_rental_model.json")
        return m
    return joblib.load("champion_bike_rental_model.pkl")


model = load_model()

# ---- Encodings: identical to the notebook's LabelEncoder (alphabetical on strings) ----
SEASON = {"Fall": 0, "Spring": 1, "Summer": 2, "Winter": 3}   # dataset calls spring "springer"
YEAR = {"2011": 0, "2012": 1}
HOLIDAY = {"No": 0, "Yes": 1}
WORKINGDAY = {"Non-working day": 0, "Working day": 1}
WEATHER = {"Clear": 0, "Heavy Rain": 1, "Light Snow": 2, "Mist": 3}
# NOTE: month was encoded as a *string*, so the order is '1','10','11','12','2',...,'9'
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_CODE = {name: sorted(str(m) for m in range(1, 13)).index(str(i + 1)) for i, name in enumerate(MONTH_NAMES)}
WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
FEATURES = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "temp", "hum", "windspeed"]


def build_row(year, month, weekday, hour, season, holiday, workingday, weather, temp_c, hum_pct, wind_kmh):
    return {
        "season": SEASON[season], "yr": YEAR[year], "mnth": MONTH_CODE[month], "hr": hour,
        "holiday": HOLIDAY[holiday], "weekday": WEEKDAYS.index(weekday),
        "workingday": WORKINGDAY[workingday], "weathersit": WEATHER[weather],
        "temp": temp_c / 41,        # dataset stores temp normalised by 41 (max)
        "hum": hum_pct / 100, "windspeed": wind_kmh / 67,
    }


def predict(rows):
    df = pd.DataFrame(rows)[FEATURES]
    return [max(0, round(float(p))) for p in model.predict(df)]


# ------------------------------------------------------------------ themes
THEMES = {
    "🏙️ City Ride": dict(bg="linear-gradient(160deg,#0f2027 0%,#203a43 50%,#2c5364 100%)", accent="#38bdf8",
                         text="#e8f1f5", card="rgba(255,255,255,0.08)", sky=("#1e3c52", "#38bdf8"), sun="#fde68a", bldg="#0b1c26", dark=True),
    "🌳 Sunny Park": dict(bg="linear-gradient(160deg,#fef9c3 0%,#bbf7d0 55%,#86efac 100%)", accent="#16a34a",
                          text="#14352a", card="rgba(255,255,255,0.65)", sky=("#fde68a", "#86efac"), sun="#f59e0b", bldg="#4d7c5a", dark=False),
    "🌙 Night Mode": dict(bg="linear-gradient(160deg,#0b1026 0%,#1a1446 55%,#2b1055 100%)", accent="#a78bfa",
                          text="#ece9ff", card="rgba(255,255,255,0.07)", sky=("#1a1446", "#6d28d9"), sun="#e0e7ff", bldg="#080a1c", dark=True),
    "🌧️ Rainy Day": dict(bg="linear-gradient(160deg,#cbd5e1 0%,#94a3b8 55%,#64748b 100%)", accent="#0ea5e9",
                         text="#0f172a", card="rgba(255,255,255,0.55)", sky=("#94a3b8", "#cbd5e1"), sun="#e2e8f0", bldg="#334155", dark=False),
}


def hero_svg(t):
    """Inline skyline + cyclist illustration, so it never fails to load (no external images)."""
    s1, s2 = t["sky"]
    bld = "".join(
        f'<rect x="{x}" y="{240 - h}" width="{w}" height="{h}" fill="{t["bldg"]}" opacity="0.9"/>'
        for x, w, h in [(20, 50, 90), (80, 40, 140), (130, 60, 110), (200, 45, 170), (255, 55, 100),
                        (320, 40, 150), (370, 65, 120), (445, 45, 180), (500, 60, 95), (570, 50, 140),
                        (630, 60, 115), (700, 45, 160), (755, 60, 100), (825, 50, 135), (885, 60, 105), (950, 40, 150)]
    )
    wins = "".join(
        f'<rect x="{x}" y="{y}" width="5" height="7" fill="#fde68a" opacity="0.55"/>'
        for x in (92, 213, 333, 458, 583, 713, 838) for y in (115, 140, 165, 190)
    )
    acc = t["accent"]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 300" preserveAspectRatio="xMidYMid slice">
<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{s1}"/><stop offset="1" stop-color="{s2}"/></linearGradient></defs>
<rect width="1000" height="300" fill="url(#g)"/>
<circle cx="560" cy="62" r="30" fill="{t['sun']}" opacity="0.95"/>
{bld}{wins}
<rect y="240" width="1000" height="60" fill="{t['bldg']}"/>
<rect y="262" width="1000" height="4" fill="{acc}" opacity="0.6"/>
<g transform="translate(735,135)" stroke="{acc}" stroke-width="5" fill="none" stroke-linecap="round" stroke-linejoin="round">
<circle cx="0" cy="75" r="30" /><circle cx="110" cy="75" r="30"/>
<path d="M0 75 L35 30 L85 30 L110 75 M35 30 L55 75 L0 75 M55 75 L85 30"/>
<path d="M30 22 L48 22 M85 30 L92 10 L108 10"/>
<circle cx="52" cy="2" r="9" fill="{acc}"/><path d="M52 12 L45 28 L58 50"/>
</g></svg>'''
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def inject_css(t):
    hero_bg = hero_svg(t)
    custom = f"assets/{t_key_slug}.jpg"
    if os.path.exists(custom):   # optional: drop your own photo in assets/ (e.g. assets/city.jpg)
        with open(custom, "rb") as f:
            hero_bg = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Poppins', sans-serif; }}
.stApp {{ background: {t['bg']}; background-attachment: fixed; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; max-width: 1150px; }}
.stApp, .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp li,
[data-testid="stWidgetLabel"] p, [data-testid="stMetricValue"], [data-testid="stCaptionContainer"] {{ color: {t['text']} !important; }}
section[data-testid="stSidebar"] {{ background: {t['card']}; backdrop-filter: blur(14px); }}
.hero {{ position: relative; border-radius: 22px; overflow: hidden; height: 270px; margin-bottom: 22px;
        background: url('{hero_bg}') center/cover; box-shadow: 0 12px 40px rgba(0,0,0,.28); }}
.hero::after {{ content:''; position:absolute; inset:0; background: linear-gradient(180deg, rgba(0,0,0,.05), rgba(0,0,0,.55)); }}
.hero-text {{ position:absolute; z-index:2; left:32px; bottom:24px; right:32px; }}
.stApp .hero-text h1 {{ color:#fff !important; font-size: 2.3rem; margin:0; font-weight:700; text-shadow: 0 2px 12px rgba(0,0,0,.5); }}
.stApp .hero-text h1 span, .stApp .hero-text .pill {{ color:#fff !important; }}
.stApp .hero-text p {{ color:#f1f5f9 !important; margin:6px 0 0; font-size:1.02rem; text-shadow: 0 1px 8px rgba(0,0,0,.5); }}
.pill {{ display:inline-block; background:{t['accent']}; color:#fff; padding:3px 12px; border-radius:999px; font-size:.75rem; font-weight:600; margin-bottom:8px; }}
.card-title {{ font-weight:600; font-size:1.05rem; margin-bottom:4px; color:{t['text']}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:{t['card']}; backdrop-filter: blur(12px);
        border-radius:18px !important; border:1px solid rgba(255,255,255,.18) !important; box-shadow:0 8px 28px rgba(0,0,0,.14); }}
div[data-baseweb="select"] > div {{ background: rgba(255,255,255,.92) !important; border-radius:10px; }}
div[data-baseweb="select"] * {{ color:#0f172a !important; }}
.stButton > button {{ width:100%; background:{t['accent']}; color:#fff; border:none; border-radius:12px; padding:.75rem 1rem;
        font-weight:600; font-size:1.05rem; box-shadow:0 6px 18px rgba(0,0,0,.25); transition: transform .15s; }}
.stButton > button:hover, .stButton > button:focus, .stButton > button:active {{ background:{t['accent']} !important; color:#fff !important; border:none !important; filter:brightness(1.12); transform: translateY(-2px); }}
.stApp .stButton button p {{ color:#fff !important; }}
.result {{ text-align:center; padding: 8px 0 2px; }}
.result .num {{ font-size:4.2rem; font-weight:700; line-height:1; color:{t['accent']}; }}
.result .unit {{ font-size:1rem; opacity:.85; }}
.badge {{ display:inline-block; padding:5px 16px; border-radius:999px; color:#fff; font-weight:600; margin:10px 0 4px; }}
.bar {{ height:12px; background:rgba(128,128,128,.28); border-radius:999px; overflow:hidden; margin:10px 0 4px; }}
.bar > div {{ height:100%; border-radius:999px; }}
.footer {{ text-align:center; opacity:.75; font-size:.85rem; margin-top:26px; }}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🎨 Theme")
    theme_name = st.radio("Choose a look", list(THEMES), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("Predicts hourly bike rental demand from time, calendar and weather features.")
    st.caption("**Model:** tuned XGBoost  \n**R²:** 0.9535  \n**RMSE:** ≈ 38 bikes  \n**Data:** 2011–2012 hourly bike sharing, 17,379 records")

theme = THEMES[theme_name]
t_key_slug = {"🏙️ City Ride": "city", "🌳 Sunny Park": "park", "🌙 Night Mode": "night", "🌧️ Rainy Day": "rain"}[theme_name]
inject_css(theme)

# ------------------------------------------------------------------ hero
st.markdown("""
<div class="hero"><div class="hero-text">
<span class="pill">🚲 Smart Bike Sharing</span>
<h1>Bike Rental Demand Predictor</h1>
<p>Forecast hourly demand so operators can keep the right number of bikes in the right place.</p>
</div></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ inputs
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown('<div class="card-title">📅 Date & Time</div>', unsafe_allow_html=True)
        year = st.selectbox("Year", list(YEAR), index=1)
        month = st.selectbox("Month", MONTH_NAMES, index=5)
        weekday = st.selectbox("Day of week", WEEKDAYS, index=2)
        hour = st.slider("Hour of day", 0, 23, 8, format="%d:00")
with c2:
    with st.container(border=True):
        st.markdown('<div class="card-title">🗓️ Day Type</div>', unsafe_allow_html=True)
        season = st.selectbox("Season", list(SEASON), index=2)
        holiday = st.selectbox("Holiday?", list(HOLIDAY))
        workingday = st.selectbox("Day type", list(WORKINGDAY), index=1)
with c3:
    with st.container(border=True):
        st.markdown('<div class="card-title">⛅ Weather</div>', unsafe_allow_html=True)
        weather = st.selectbox("Conditions", list(WEATHER))
        temp_c = st.slider("Temperature (°C)", 0, 41, 20)
        hum_pct = st.slider("Humidity (%)", 0, 100, 60)
        wind_kmh = st.slider("Wind speed (km/h)", 0, 67, 12)

st.write("")
go = st.button("🔮 Predict demand")

# ------------------------------------------------------------------ result
if go:
    args = (year, month, weekday, hour, season, holiday, workingday, weather, temp_c, hum_pct, wind_kmh)
    pred = predict([build_row(*args)])[0]

    if pred < 60:
        level, col = "Low demand", "#22c55e"
    elif pred < 180:
        level, col = "Moderate demand", "#eab308"
    elif pred < 320:
        level, col = "High demand", "#f97316"
    else:
        level, col = "Very high demand", "#ef4444"
    pct = min(100, round(pred / 600 * 100))

    left, right = st.columns([1, 1.5])
    with left:
        with st.container(border=True):
            st.markdown(f"""<div class="result">
<div class="unit">Predicted rentals at {hour}:00</div>
<div class="num">{pred}</div><div class="unit">bikes</div>
<span class="badge" style="background:{col}">{level}</span>
<div class="bar"><div style="width:{pct}%; background:{col}"></div></div>
<div class="unit">Expected range: {max(0, pred - 38)} – {pred + 38} bikes (±38 typical error)</div>
</div>""", unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown('<div class="card-title">📈 Demand across the day (same conditions)</div>', unsafe_allow_html=True)
            day_rows = [build_row(year, month, weekday, h, season, holiday, workingday, weather, temp_c, hum_pct, wind_kmh) for h in range(24)]
            curve = pd.DataFrame({"Hour": list(range(24)), "Predicted bikes": predict(day_rows)})
            sel = curve[curve["Hour"] == hour]
            base = alt.Chart(curve).encode(x=alt.X("Hour:Q", scale=alt.Scale(domain=[0, 23]), axis=alt.Axis(tickMinStep=1)),
                                           y=alt.Y("Predicted bikes:Q"))
            chart = (base.mark_area(color=theme["accent"], opacity=0.35, line={"color": theme["accent"], "strokeWidth": 3})
                     + alt.Chart(sel).mark_point(color="#ef4444", size=140, filled=True).encode(x="Hour:Q", y="Predicted bikes:Q")
                     ).properties(height=230, background="transparent").configure_axis(
                labelColor=theme["text"], titleColor=theme["text"], gridColor="rgba(128,128,128,0.25)", domainColor="rgba(128,128,128,0.4)"
            ).configure_view(strokeWidth=0)
            st.altair_chart(chart, use_container_width=True, theme=None)
            curve = curve.set_index("Hour")
            peak = int(curve["Predicted bikes"].idxmax())
            st.caption(f"Peak demand under these conditions is around {peak}:00 ({int(curve['Predicted bikes'].max())} bikes).")

st.markdown('<div class="footer">Built with Streamlit & XGBoost · Trained on 2011–2012 hourly bike sharing data (year limited to 2011/2012)</div>', unsafe_allow_html=True)
