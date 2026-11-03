import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import re
import requests
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# 1. Page configuration (unchanged)
# ------------------------------------------------------------------
st.set_page_config(page_title="Telangana EV & AQI Dashboard", page_icon="⚡", layout="wide")

# ------------------------------------------------------------------
# 2. Data loading (original logic + safe Date parsing)
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Telangana_EV_AQI_Dataset.xlsx")
    d = pd.read_excel(file_path, sheet_name="Raw Data")
    d.columns = [str(c).strip() for c in d.columns]
    if "Date" in d.columns:
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        d = d.sort_values([c for c in ["Zone", "Date"] if c in d.columns])
    for c in ("Zone", "City", "State"):
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip().str.title()
    return d.drop_duplicates()


def find_col(d, candidates):
    lower = {c.lower(): c for c in d.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading Excel file: {e}")
    st.stop()

# Column mapping: uses real columns when present, never invents any
GEO = find_col(df, ["City", "Zone"])                       # geographic unit
STATE = find_col(df, ["State"])
EV_COUNT = find_col(df, ["EV Count", "Number of EVs", "EVs", "EV Registrations", "Total EVs", "EV Population"])
POP = find_col(df, ["Population"])
EV_COL = EV_COUNT or find_col(df, ["Charging Sessions"])  # proxy if no count
EV_LABEL = "Electric Vehicles" if EV_COUNT else "Charging Sessions (EV activity proxy)"
EV_KIND = "count" if EV_COUNT else "proxy"

# ------------------------------------------------------------------
# 3. Authentication (simple session-based)
# ------------------------------------------------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_login(name, email, password):
    errors = []
    if len(name.strip()) < 2:
        errors.append("Please enter your name (at least 2 characters).")
    if not EMAIL_RE.match(email.strip()):
        errors.append("Please enter a valid email address.")
    if len(password) < 6:
        errors.append("Password must be at least 6 characters.")
    required = st.secrets.get("APP_PASSWORD", None) if hasattr(st, "secrets") else None
    if required and password != required:
        errors.append("Invalid password.")
    return errors


def login_page():
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("## ⚡ Telangana EV & AQI Dashboard")
        st.caption("Sign in to personalise the dashboard for your location.")
        with st.form("login_form"):
            name = st.text_input("Name", value=st.session_state.get("remembered_name", ""))
            email = st.text_input("Email ID", value=st.session_state.get("remembered_email", ""))
            password = st.text_input("Password", type="password")
            remember = st.checkbox("Remember me (this browser session)")
            submitted = st.form_submit_button("Login", use_container_width=True)
        if submitted:
            errs = validate_login(name, email, password)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                st.session_state.update(logged_in=True, user_name=name.strip(), user_email=email.strip())
                if remember:
                    st.session_state.update(remembered_name=name.strip(), remembered_email=email.strip())
                st.rerun()


if not st.session_state.get("logged_in"):
    login_page()
    st.stop()

# ------------------------------------------------------------------
# 4. AQI helpers
# ------------------------------------------------------------------
CPCB = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very Poor"), (10**9, "Severe")]
EPA = [(50, "Good"), (100, "Moderate"), (150, "Unhealthy for Sensitive Groups"), (200, "Unhealthy"),
       (300, "Very Unhealthy"), (10**9, "Hazardous")]


def aqi_category(v, bands=CPCB):
    if v is None or pd.isna(v):
        return "Data unavailable"
    return next(label for limit, label in bands if v <= limit)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_live_aqi(city):
    """Live AQI via WAQI (token in st.secrets['WAQI_TOKEN']). Returns dict or None."""
    try:
        token = st.secrets.get("WAQI_TOKEN", None)
    except Exception:
        token = None
    if not token:
        return None
    try:
        r = requests.get(f"https://api.waqi.info/feed/{city}/", params={"token": token}, timeout=5)
        j = r.json()
        if j.get("status") == "ok" and str(j["data"].get("aqi", "")).lstrip("-").isdigit():
            return {"aqi": float(j["data"]["aqi"]), "time": j["data"].get("time", {}).get("s", "n/a")}
    except Exception:
        return None
    return None


# ------------------------------------------------------------------
# 5. Metric helpers (all values come from the dataset)
# ------------------------------------------------------------------
def city_frame(city, years):
    return df[(df[GEO] == city) & (df["Year"].isin(years))]


def yearly(cf):
    if cf.empty:
        return pd.DataFrame(columns=["Year", "AQI", "EV"])
    agg = (lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan) if EV_KIND == "count" else "sum"
    return cf.groupby("Year").agg(AQI=("AQI Index", "mean"), EV=(EV_COL, agg)).reset_index()


def city_metrics(city, years):
    cf = city_frame(city, years)
    m = {"city": city, "empty": cf.empty, "aqi": None, "aqi_when": None, "ev": None, "growth": None,
         "pop": None, "rate": None, "yearly": yearly(cf), "frame": cf}
    if cf.empty:
        return m
    latest = cf.dropna(subset=["AQI Index"]).tail(1)
    if not latest.empty:
        m["aqi"] = float(latest["AQI Index"].iloc[0])
        m["aqi_when"] = latest["Date"].iloc[0].strftime("%b %Y") if "Date" in latest and pd.notna(latest["Date"].iloc[0]) else f"{int(latest['Year'].iloc[0])}"
    y = m["yearly"].dropna(subset=["EV"])
    if not y.empty:
        m["ev"] = float(y["EV"].iloc[-1])
        if len(y) >= 2 and y["EV"].iloc[0] > 0:
            m["growth"] = (y["EV"].iloc[-1] / y["EV"].iloc[0] - 1) * 100
    if POP and cf[POP].notna().any():
        m["pop"] = float(cf[POP].dropna().iloc[-1])
        if EV_KIND == "count" and m["ev"] is not None and m["pop"] > 0:
            m["rate"] = m["ev"] / m["pop"] * 1000
    return m


def fmt(v, spec="{:,.0f}", suffix=""):
    return "Data unavailable" if v is None or pd.isna(v) else spec.format(v) + suffix


def corr_note(m):
    d = m["frame"].dropna(subset=["AQI Index", EV_COL])
    if len(d) < 4 or d["AQI Index"].std() == 0 or d[EV_COL].std() == 0:
        return None
    return d["AQI Index"].corr(d[EV_COL])


# ------------------------------------------------------------------
# 6. Sidebar: profile, location, comparison, original filters
# ------------------------------------------------------------------
st.sidebar.title("⚡ Dashboard Controls")
with st.sidebar.expander("👤 User Profile", expanded=True):
    st.write(f"**Name:** {st.session_state['user_name']}")
    st.write(f"**Email:** {st.session_state['user_email']}")
    if st.button("Logout"):
        for k in ("logged_in", "user_name", "user_email"):
            st.session_state.pop(k, None)
        st.rerun()

with st.sidebar.expander("📍 My Location", expanded=True):
    states = sorted(df[STATE].dropna().unique()) if STATE else ["Telangana"]
    sel_state = st.selectbox("State", states)
    pool = df[df[STATE] == sel_state] if STATE else df
    cities = sorted(pool[GEO].dropna().unique())
    sel_city = st.selectbox("City", cities) if cities else None
    sel_area = st.text_input("Area / locality (optional)", placeholder="e.g. Gachibowli")

all_years = sorted(df["Year"].dropna().unique().tolist())
with st.sidebar.expander("🔍 Comparison", expanded=True):
    c1_default = cities.index(sel_city) if sel_city in cities else 0
    city1 = st.selectbox("City 1", cities, index=c1_default, key="c1")
    others = [c for c in cities if c != city1] or cities
    city2 = st.selectbox("City 2", cities, index=cities.index(others[0]) if others else 0, key="c2")
    p_years = st.multiselect("Time period (new sections)", all_years, default=all_years)

st.sidebar.markdown("### Original dashboard filters")
years = sorted(df["Year"].unique().tolist())
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)
zones = sorted(df["Zone"].unique().tolist())
selected_zones = st.sidebar.multiselect("Select Zone(s)", zones, default=zones[:4])
filtered_df = df[(df["Year"].isin(selected_years)) & (df["Zone"].isin(selected_zones))]

with st.sidebar.expander("🧾 Detected data columns"):
    st.write({"Location column": GEO, "State column": STATE or "not found (assumed Telangana)",
              "EV column": f"{EV_COL} ({'real count' if EV_COUNT else 'proxy'})",
              "Population": POP or "not found", "All columns": list(df.columns)})

# ------------------------------------------------------------------
# 7. Original dashboard (code preserved; st.stop -> return)
# ------------------------------------------------------------------
def render_original():
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg AQI Index", f"{filtered_df['AQI Index'].mean():.1f}")
    col2.metric("Total EV Power", f"{filtered_df['EV Power Consumed (kWh)'].sum():,.0f} kWh")
    col3.metric("Total Charging Sessions", f"{filtered_df['Charging Sessions'].sum():,.0f}")
    col4.metric("Active Zones Monitored", len(filtered_df["Zone"].unique()))
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["💨 Air Quality Trends", "🔌 EV Infrastructure", "📋 Raw Data"])
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Monthly AQI Trend by Zone")
            st.plotly_chart(px.line(filtered_df, x="Date", y="AQI Index", color="Zone", markers=True), use_container_width=True)
        with c2:
            st.subheader("PM2.5 vs PM10 Concentration")
            st.plotly_chart(px.scatter(filtered_df, x="PM2.5 (µg/m³)", y="PM10 (µg/m³)", color="Zone", size="AQI Index"), use_container_width=True)
    with tab2:
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Power Consumption by Zone")
            df_power = filtered_df.groupby("Zone")["EV Power Consumed (kWh)"].sum().reset_index()
            st.plotly_chart(px.bar(df_power, x="Zone", y="EV Power Consumed (kWh)", color="EV Power Consumed (kWh)"), use_container_width=True)
        with c4:
            st.subheader("Charging Sessions Over Time")
            st.plotly_chart(px.area(filtered_df, x="Date", y="Charging Sessions", color="Zone"), use_container_width=True)
    with tab3:
        st.subheader("Filtered Dataset View")
        st.dataframe(filtered_df, use_container_width=True)


# ------------------------------------------------------------------
# 8. New personalised sections
# ------------------------------------------------------------------
def kpi_cards(m, live):
    a, b, c = st.columns(3)
    with a, st.container(border=True):
        st.markdown(f"**💨 {m['city']} AQI**")
        if live:
            st.metric("Live AQI (WAQI)", f"{live['aqi']:.0f}", aqi_category(live["aqi"], EPA))
            st.caption(f"🟢 Live · updated {live['time']} · US-EPA scale")
        elif m["aqi"] is not None:
            st.metric("Latest dataset AQI", f"{m['aqi']:.0f}", aqi_category(m["aqi"]), delta_color="off")
            st.caption(f"🗂️ Historical dataset · {m['aqi_when']} · CPCB bands. Live AQI needs a WAQI_TOKEN secret.")
        else:
            st.info("AQI data is currently unavailable for this location.")
    with b, st.container(border=True):
        st.markdown(f"**🚗 {EV_LABEL}**")
        if m["ev"] is not None:
            st.metric("Latest value", fmt(m["ev"]), None)
            st.caption("🗂️ Dataset" + ("" if EV_COUNT else " · no EV-count column found, showing charging sessions"))
        else:
            st.info("EV data unavailable for this location.")
    with c, st.container(border=True):
        st.markdown("**📈 EV Growth**")
        st.metric("First → last selected year", fmt(m["growth"], "{:+.1f}", "%"))
        st.caption("🧮 Calculated from yearly dataset values")


def analysis_chart(m):
    y = m["yearly"]
    if y.empty or y["AQI"].isna().all():
        st.info("No historical data available for this location.")
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=y["Year"], y=y["AQI"], name="Avg AQI", mode="lines+markers"), secondary_y=False)
    fig.add_trace(go.Bar(x=y["Year"], y=y["EV"], name=EV_LABEL, opacity=0.35), secondary_y=True)
    fig.update_xaxes(title_text="Year", type="category")
    fig.update_yaxes(title_text="Average AQI", secondary_y=False)
    fig.update_yaxes(title_text=EV_LABEL, secondary_y=True)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    r = corr_note(m)
    if r is not None:
        st.caption(f"Observed correlation (AQI vs {EV_COL}, monthly records): r = {r:.2f}. "
                   "This is an association in the data, not proof that EVs caused a change in AQI.")
    else:
        st.caption("Not enough data points to compute a correlation.")


def compare_table(m1, m2):
    rows = [("AQI (latest dataset)", fmt(m1["aqi"], "{:.0f}"), fmt(m2["aqi"], "{:.0f}")),
            ("AQI Category", aqi_category(m1["aqi"]), aqi_category(m2["aqi"])),
            (EV_LABEL, fmt(m1["ev"]), fmt(m2["ev"])),
            ("EV Growth", fmt(m1["growth"], "{:+.1f}", "%"), fmt(m2["growth"], "{:+.1f}", "%"))]
    if POP:
        rows.append(("Population", fmt(m1["pop"]), fmt(m2["pop"])))
    if POP and EV_COUNT:
        rows.append(("EVs per 1,000 people", fmt(m1["rate"], "{:.1f}"), fmt(m2["rate"], "{:.1f}")))
    st.dataframe(pd.DataFrame(rows, columns=["Metric", m1["city"], m2["city"]]), hide_index=True, use_container_width=True)


def compare_charts(m1, m2):
    t1, t2, t3 = st.tabs(["AQI", EV_LABEL, "Trends"])
    names = [m1["city"], m2["city"]]
    for tab, key, label in ((t1, "aqi", "AQI"), (t2, "ev", EV_LABEL)):
        with tab:
            vals = [m1[key], m2[key]]
            if all(v is None for v in vals):
                st.info("Data unavailable for both cities.")
                continue
            fig = px.bar(x=names, y=vals, color=names, labels={"x": "City", "y": label}, text=[fmt(v) for v in vals])
            fig.update_layout(showlegend=True, legend_title_text="City")
            st.plotly_chart(fig, use_container_width=True)
    with t3:
        both = pd.concat([m1["yearly"].assign(City=m1["city"]), m2["yearly"].assign(City=m2["city"])])
        if both.empty:
            st.info("No historical data available.")
        else:
            for col, label in (("AQI", "Average AQI"), ("EV", EV_LABEL)):
                fig = px.line(both, x="Year", y=col, color="City", markers=True, labels={col: label})
                fig.update_xaxes(type="category")
                fig.update_layout(hovermode="x unified", title=f"{label} over time")
                st.plotly_chart(fig, use_container_width=True)


def build_insights(m1, m2):
    out, a, b = [], m1, m2
    if a["aqi"] is not None and b["aqi"] is not None:
        lo, hi = (a, b) if a["aqi"] < b["aqi"] else (b, a)
        if a["aqi"] == b["aqi"]:
            out.append("Both cities have the same latest recorded AQI.")
        else:
            out.append(f"{lo['city']} has the lower latest recorded AQI ({lo['aqi']:.0f} vs {hi['aqi']:.0f}, a difference of {hi['aqi'] - lo['aqi']:.0f} points).")
    if a["ev"] is not None and b["ev"] is not None:
        hi = a if a["ev"] >= b["ev"] else b
        out.append(f"{hi['city']} has higher {EV_LABEL.lower()} in the selected period.")
        if all(x["aqi"] is not None for x in (a, b)) and (a["ev"] - b["ev"]) * (b["aqi"] - a["aqi"]) > 0:
            out.append(f"{hi['city']} has higher EV activity and a lower recorded AQI than the other city. This shows an association in the available data, but it does not by itself establish causation.")
    for m in (a, b):
        if m["growth"] is not None:
            out.append(f"{m['city']}: {EV_LABEL.lower()} changed by {m['growth']:+.1f}% between the first and last selected year.")
        r = corr_note(m)
        if r is not None:
            out.append(f"{m['city']}: AQI and {EV_COL} have a correlation of {r:.2f} in monthly records (association only).")
    if not out:
        out.append("Not enough data to generate insights for the selected cities and period.")
    return out


def render_personalised():
    if not sel_city:
        st.warning("Please select a location in the sidebar.")
        return
    if not p_years:
        st.warning("Select at least one year in the sidebar.")
        return
    area = f" · {sel_area.strip()}" if sel_area.strip() else ""
    st.markdown(f"### 📍 Your Selected Location: {sel_city}, {sel_state}{area}")
    st.caption(f"Welcome, {st.session_state['user_name']}. Area is shown for reference only; the dataset has no area-level data.")
    me = city_metrics(sel_city, p_years)
    if me["empty"]:
        st.warning("No data available for this location.")
    kpi_cards(me, fetch_live_aqi(sel_city))

    st.markdown("### 🧠 Personalized AQI + EV Analysis")
    analysis_chart(me)

    st.markdown("### 🔍 Compare Two Cities")
    if city1 == city2:
        st.warning("Please choose two different cities to compare.")
        m1 = m2 = None
    else:
        m1, m2 = city_metrics(city1, p_years), city_metrics(city2, p_years)
        if m1["empty"] or m2["empty"]:
            st.warning("No data available for one of the selected cities in this period.")
            m1 = m2 = None
    if m1:
        left, right = st.columns(2)
        for col, m in ((left, m1), (right, m2)):
            with col, st.container(border=True):
                st.markdown(f"**{m['city']}**")
                st.metric("AQI", fmt(m["aqi"], "{:.0f}"), aqi_category(m["aqi"]), delta_color="off")
                st.metric(EV_LABEL, fmt(m["ev"]))
        compare_table(m1, m2)
        compare_charts(m1, m2)

    st.markdown("### 📊 Your Location Summary")
    with st.container(border=True):
        st.markdown(f"**Location:** {sel_city}, {sel_state}  \n"
                    f"**AQI:** {fmt(me['aqi'], '{:.0f}')}  \n"
                    f"**AQI Status:** {aqi_category(me['aqi'])}  \n"
                    f"**{EV_LABEL}:** {fmt(me['ev'])}  \n"
                    f"**EV Growth:** {fmt(me['growth'], '{:+.1f}', '%')}  \n"
                    f"**Compared with:** {city2 if city1 == sel_city else city1 if city2 == sel_city else f'{city1} and {city2}'}")
        st.caption("Values are from the dataset unless marked live above; growth is calculated.")

    st.markdown("### 💡 Location Insights")
    if m1:
        for line in build_insights(m1, m2):
            st.markdown(f"- {line}")
    else:
        st.info("Select two different cities with data to generate insights.")


# ------------------------------------------------------------------
# 9. Page layout
# ------------------------------------------------------------------
st.title("🌱 Telangana Urban AQI & EV Infrastructure Dashboard")
st.caption("Analyzing regional air quality metrics and clean energy adoption.")

tab_me, tab_orig = st.tabs(["📍 My Location & Comparison", "📊 Original Dashboard"])
with tab_me:
    render_personalised()
with tab_orig:
    render_original()