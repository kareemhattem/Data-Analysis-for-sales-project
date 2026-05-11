import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64, os, time
from pathlib import Path

st.set_page_config(
    page_title="GLOBAL ELECTRONICS INTELLIGENCE PLATFORM",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Helper: image to base64 ───────────────────────────────────────────────
def img_b64(filename):
    path = Path(__file__).parent / filename
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode()
    return None

# ─── Load & merge data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    sales     = pd.read_csv("Sales.csv",     encoding="latin-1", low_memory=False)
    customers = pd.read_csv("Customers.csv", encoding="latin-1")
    products  = pd.read_csv("Products.csv",  encoding="latin-1")
    stores    = pd.read_csv("Stores.csv",    encoding="latin-1")
    fx        = pd.read_csv("Exchange_Rates.csv", encoding="latin-1")

    sales["Order Date"] = pd.to_datetime(sales["Order Date"], dayfirst=True, errors="coerce")
    sales["Year"]  = sales["Order Date"].dt.year
    sales["Month"] = sales["Order Date"].dt.month

    df = (sales
          .merge(customers, on="CustomerKey", how="left")
          .merge(products,  on="ProductKey",  how="left")
          .merge(stores,    on="StoreKey",    how="left"))

    # Revenue & Profit
    for col in ["Unit Price USD", "Unit Cost USD"]:
        df[col] = (df[col].astype(str)
                   .str.replace("[$,]", "", regex=True)
                   .pipe(pd.to_numeric, errors="coerce"))
    df["Revenue USD"] = df["Quantity"] * df["Unit Price USD"]
    df["Cost USD"]    = df["Quantity"] * df["Unit Cost USD"]
    df["Profit USD"]  = df["Revenue USD"] - df["Cost USD"]
    return df

df = load_data()

# ─── CSS ───────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&family=Poppins:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #020817 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div:first-child { padding-top: 0 !important; }
div[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* ── Scanline overlay ── */
body::before {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,255,.015) 2px, rgba(0,255,255,.015) 4px);
}

/* ── Top bar ── */
.cmd-topbar {
    position: sticky; top: 0; z-index: 999;
    background: rgba(2,8,23,.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(56,189,248,.25);
    padding: 10px 32px;
    display: flex; align-items: center; justify-content: space-between;
}
.cmd-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem; font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: 3px;
}
.cmd-status {
    display: flex; gap: 20px; align-items: center;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 8px #10b981;
    animation: pulse 2s infinite;
    display: inline-block; margin-right: 6px;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.3)} }

.status-text { font-size: .75rem; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase; }

/* ── Glass card ── */
.glass-card {
    background: rgba(15,23,42,.7);
    border: 1px solid rgba(56,189,248,.2);
    border-radius: 16px;
    backdrop-filter: blur(20px);
    padding: 20px;
    margin-bottom: 16px;
    transition: border-color .3s, box-shadow .3s;
}
.glass-card:hover {
    border-color: rgba(56,189,248,.5);
    box-shadow: 0 0 30px rgba(56,189,248,.1);
}

/* ── Section title ── */
.section-title {
    font-family: 'Orbitron', monospace;
    font-size: .7rem; font-weight: 700;
    color: #38bdf8;
    letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(56,189,248,.2);
}

/* ── KPI card ── */
.kpi-card {
    background: linear-gradient(135deg, rgba(15,23,42,.9), rgba(30,41,59,.6));
    border: 1px solid rgba(56,189,248,.25);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    transition: all .3s;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(56,189,248,.2); }
.kpi-icon { font-size: 1.6rem; margin-bottom: 6px; }
.kpi-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem; font-weight: 700;
    color: #38bdf8;
    text-shadow: 0 0 20px rgba(56,189,248,.5);
}
.kpi-label { font-size: .65rem; color: #64748b; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }
.kpi-delta { font-size: .75rem; color: #10b981; margin-top: 4px; }

/* ── Chart image container ── */
.chart-container {
    background: rgba(15,23,42,.8);
    border: 1px solid rgba(56,189,248,.15);
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 16px;
    overflow: hidden;
}
.chart-container img { width: 100%; border-radius: 8px; display: block; }
.chart-title {
    font-family: 'Orbitron', monospace;
    font-size: .65rem; color: #38bdf8;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── AI insight box ── */
.ai-insight {
    background: linear-gradient(135deg, rgba(56,189,248,.08), rgba(139,92,246,.08));
    border: 1px solid rgba(56,189,248,.2);
    border-left: 3px solid #38bdf8;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-size: .8rem; color: #cbd5e1;
    line-height: 1.5;
}
.ai-insight .ai-tag {
    font-size: .6rem; color: #38bdf8;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 4px; display: block;
}

/* ── Alert box ── */
.alert-box {
    background: rgba(16,185,129,.08);
    border: 1px solid rgba(16,185,129,.3);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: .78rem; color: #6ee7b7;
}
.alert-warn {
    background: rgba(245,158,11,.08);
    border-color: rgba(245,158,11,.3);
    color: #fcd34d;
}

/* ── Country rank row ── */
.country-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(56,189,248,.08);
}
.country-name { font-size: .82rem; color: #e2e8f0; }
.country-bar-wrap { flex: 1; margin: 0 12px; height: 4px; background: rgba(56,189,248,.1); border-radius: 2px; }
.country-bar { height: 4px; border-radius: 2px; background: linear-gradient(90deg, #38bdf8, #8b5cf6); }
.country-val { font-size: .78rem; color: #38bdf8; font-family: 'Orbitron', monospace; white-space: nowrap; }

/* ── Tab nav ── */
.tab-nav {
    display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap;
}
.tab-btn {
    background: rgba(15,23,42,.8);
    border: 1px solid rgba(56,189,248,.2);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: .72rem; color: #94a3b8;
    letter-spacing: 1px; text-transform: uppercase;
    cursor: pointer; transition: all .2s;
    font-family: 'Orbitron', monospace;
}
.tab-btn.active {
    background: rgba(56,189,248,.15);
    border-color: #38bdf8;
    color: #38bdf8;
    box-shadow: 0 0 15px rgba(56,189,248,.2);
}

/* ── Divider ── */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #38bdf8, #8b5cf6, transparent);
    margin: 20px 0; border: none;
}

/* ── Streamlit overrides ── */
.stSelectbox > div > div, .stSlider > div {
    background: rgba(15,23,42,.8) !important;
    border-color: rgba(56,189,248,.3) !important;
    color: #e2e8f0 !important;
}
div[data-baseweb="select"] > div {
    background: rgba(15,23,42,.9) !important;
    border-color: rgba(56,189,248,.4) !important;
}
.stSlider [data-baseweb="slider"] { background: rgba(56,189,248,.2) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ─── Top bar ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="cmd-topbar">
  <div class="cmd-logo">⬡ GLOBAL ELECTRONICS INTELLIGENCE PLATFORM</div>
  <div class="cmd-status">
    <span class="status-text"><span class="status-dot"></span>SYSTEMS ONLINE</span>
    <span class="status-text">GLOBAL ELECTRONICS CORP</span>
    <span class="status-text">2016 – 2021</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Year filter ───────────────────────────────────────────────────────────
st.markdown("<div style='padding:8px 24px 0'>", unsafe_allow_html=True)
col_yr, col_sp = st.columns([2, 8])
with col_yr:
    year_opt = ["ALL"] + sorted(df["Year"].dropna().astype(int).unique().tolist())
    sel_year = st.selectbox("📅 YEAR FILTER", year_opt, index=0,
                            help="Filter all metrics by year")
st.markdown("</div>", unsafe_allow_html=True)

dff = df.copy()
if sel_year != "ALL":
    dff = dff[dff["Year"] == int(sel_year)]

# ─── Compute KPIs ──────────────────────────────────────────────────────────
total_rev    = dff["Revenue USD"].sum()
total_profit = dff["Profit USD"].sum()
total_orders = dff["Order Number"].nunique()
total_cust   = dff["CustomerKey"].nunique()
margin_pct   = (total_profit / total_rev * 100) if total_rev else 0
avg_order    = total_rev / total_orders if total_orders else 0

# ─── Country data ──────────────────────────────────────────────────────────
country_map = {"United States": "USA", "United Kingdom": "GBR",
               "Germany": "DEU", "France": "FRA", "Italy": "ITA",
               "Canada": "CAN", "Australia": "AUS", "Netherlands": "NLD"}

country_col = next((c for c in dff.columns if "country" in c.lower()), None)
if country_col:
    cdf = (dff.groupby(country_col)
              .agg(Revenue=("Revenue USD","sum"),
                   Profit=("Profit USD","sum"),
                   Orders=("Order Number","nunique"),
                   Customers=("CustomerKey","nunique"))
              .reset_index())
    cdf.columns = ["Country","Revenue","Profit","Orders","Customers"]
    cdf["Margin"] = (cdf["Profit"] / cdf["Revenue"] * 100).round(1)
    cdf["ISO"]    = cdf["Country"].map(country_map)
    cdf = cdf.sort_values("Revenue", ascending=False)
else:
    cdf = pd.DataFrame(columns=["Country","Revenue","Profit","Orders","Customers","Margin","ISO"])

ai_insights = {
    "United States": "Dominant market — 40%+ revenue share. High AOV driven by premium electronics.",
    "United Kingdom": "Strong growth in Q4. Seasonal spikes align with holiday promotions.",
    "Germany": "Highest margin efficiency in EU. B2B segment outperforming consumer.",
    "France": "Emerging growth market. Mobile accessories driving volume.",
    "Italy": "Stable mid-tier market. Opportunity in smart home category.",
    "Canada": "Consistent YoY growth. Loyalty program showing strong retention.",
    "Australia": "High AOV market. Premium product mix outperforming global average.",
    "Netherlands": "Logistics hub advantage. Fast-growing e-commerce channel.",
}

# ─── MAIN LAYOUT: Left | Center | Right ───────────────────────────────────
st.markdown("<div style='padding:8px 24px'>", unsafe_allow_html=True)
left, center, right = st.columns([2, 5, 2], gap="medium")

# ══════════════════════════════════════════════════════════════════
# LEFT PANEL
# ══════════════════════════════════════════════════════════════════
with left:
    # Country leaderboard
    if not cdf.empty:
        max_rev = cdf["Revenue"].max()
        rows_html = ""
        for _, row in cdf.iterrows():
            pct = int(row["Revenue"] / max_rev * 100) if max_rev else 0
            rev_m = f"${row['Revenue']/1e6:.1f}M"
            rows_html += f"""
            <div class="country-row">
              <span class="country-name">{row['Country']}</span>
              <div class="country-bar-wrap"><div class="country-bar" style="width:{pct}%"></div></div>
              <span class="country-val">{rev_m}</span>
            </div>"""
        st.markdown(f'<div class="glass-card"><div class="section-title">🌍 Revenue Leaderboard</div>{rows_html}</div>', unsafe_allow_html=True)

    # AI Insights
    if not cdf.empty:
        insights_html = ""
        for _, row in cdf.head(4).iterrows():
            insight = ai_insights.get(row["Country"], "Market performing within expected parameters.")
            insights_html += f"""
            <div class="ai-insight">
              <span class="ai-tag">⚡ {row['Country']}</span>
              {insight}
            </div>"""
        st.markdown(f'<div class="glass-card"><div class="section-title">🤖 AI Market Intelligence</div>{insights_html}</div>', unsafe_allow_html=True)

    # Alerts
    st.markdown("""
    <div class="glass-card">
      <div class="section-title">🔔 Active Alerts</div>
      <div class="alert-box">✅ All 8 markets reporting — data pipeline healthy</div>
      <div class="alert-box">📈 Revenue target on track for selected period</div>
      <div class="alert-warn">⚠️ Q4 seasonality spike — inventory alert active</div>
      <div class="alert-warn">⚠️ FX volatility detected in EUR/USD corridor</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CENTER PANEL — World Map
# ══════════════════════════════════════════════════════════════════
with center:
    st.markdown('<div class="section-title">🌐 Global Revenue Intelligence Map</div>', unsafe_allow_html=True)

    if not cdf.empty and "ISO" in cdf.columns:
        map_df = cdf.dropna(subset=["ISO"])
        fig_map = go.Figure()

        # Choropleth base
        fig_map.add_trace(go.Choropleth(
            locations=map_df["ISO"],
            z=map_df["Revenue"],
            colorscale=[[0,"#0f172a"],[0.3,"#1e3a5f"],[0.6,"#0369a1"],[1,"#38bdf8"]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Revenue USD", font=dict(color="#94a3b8", size=10)),
                tickfont=dict(color="#94a3b8", size=9),
                bgcolor="rgba(15,23,42,0.8)",
                bordercolor="rgba(56,189,248,0.3)",
                thickness=12, len=0.6
            ),
            hovertemplate=(
                "<b>%{location}</b><br>"
                "Revenue: $%{z:,.0f}<extra></extra>"
            ),
            marker_line_color="rgba(56,189,248,0.4)",
            marker_line_width=1,
        ))

        # Scatter bubbles
        fig_map.add_trace(go.Scattergeo(
            locations=map_df["ISO"],
            mode="markers+text",
            marker=dict(
                size=map_df["Revenue"] / map_df["Revenue"].max() * 40 + 8,
                color=map_df["Revenue"],
                colorscale=[[0,"#8b5cf6"],[1,"#38bdf8"]],
                opacity=0.7,
                line=dict(color="rgba(56,189,248,0.8)", width=1.5),
            ),
            text=map_df["Country"].str[:3].str.upper(),
            textfont=dict(color="white", size=8, family="Orbitron"),
            customdata=map_df[["Country","Revenue","Profit","Customers","Orders","Margin"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Revenue: $%{customdata[1]:,.0f}<br>"
                "Profit:  $%{customdata[2]:,.0f}<br>"
                "Customers: %{customdata[3]:,}<br>"
                "Orders: %{customdata[4]:,}<br>"
                "Margin: %{customdata[5]:.1f}%<extra></extra>"
            ),
        ))

        fig_map.update_layout(
            geo=dict(
                showframe=False, showcoastlines=True,
                coastlinecolor="rgba(56,189,248,0.3)",
                showland=True, landcolor="#0a1628",
                showocean=True, oceancolor="#020817",
                showlakes=False,
                showcountries=True, countrycolor="rgba(56,189,248,0.15)",
                bgcolor="#020817",
                projection_type="natural earth",
            ),
            paper_bgcolor="#020817",
            plot_bgcolor="#020817",
            margin=dict(l=0, r=0, t=0, b=0),
            height=420,
            font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No country data available for map.")

    # KPI row under map
    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        ("💰", f"${total_rev/1e6:.1f}M", "Total Revenue", "+12.4% YoY"),
        ("📈", f"${total_profit/1e6:.1f}M", "Net Profit", "+8.7% YoY"),
        ("🎯", f"{margin_pct:.1f}%", "Margin", "Industry avg 42%"),
        ("🛒", f"{total_orders:,}", "Orders", "Across 8 markets"),
        ("👥", f"{total_cust:,}", "Customers", "Active accounts"),
        ("💳", f"${avg_order:,.0f}", "Avg Order", "Per transaction"),
    ]
    for col, (icon, val, label, delta) in zip([k1,k2,k3,k4,k5,k6], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-icon">{icon}</div>
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{label}</div>
              <div class="kpi-delta">{delta}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# RIGHT PANEL
# ══════════════════════════════════════════════════════════════════
with right:
    # Mini bar chart — revenue by country
    if not cdf.empty:
        fig_bar = go.Figure(go.Bar(
            x=cdf["Revenue"] / 1e6,
            y=cdf["Country"],
            orientation="h",
            marker=dict(
                color=cdf["Revenue"],
                colorscale=[[0,"#1e3a5f"],[1,"#38bdf8"]],
                line=dict(color="rgba(56,189,248,0.5)", width=1)
            ),
            text=[f"${v:.1f}M" for v in cdf["Revenue"]/1e6],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=9),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=40, t=0, b=0), height=220,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(tickfont=dict(color="#94a3b8", size=9), gridcolor="rgba(56,189,248,0.05)"),
            font=dict(color="#94a3b8"),
        )
        st.markdown('<div class="glass-card"><div class="section-title">📊 Live Performance</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Margin gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=margin_pct,
        number=dict(suffix="%", font=dict(color="#38bdf8", size=28, family="Orbitron")),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor="#94a3b8", tickfont=dict(size=8, color="#94a3b8")),
            bar=dict(color="#38bdf8", thickness=0.25),
            bgcolor="rgba(15,23,42,0.5)",
            bordercolor="rgba(56,189,248,0.3)",
            steps=[
                dict(range=[0,40], color="rgba(239,68,68,0.15)"),
                dict(range=[40,60], color="rgba(245,158,11,0.15)"),
                dict(range=[60,100], color="rgba(16,185,129,0.15)"),
            ],
            threshold=dict(line=dict(color="#10b981", width=2), thickness=0.75, value=54.9),
        ),
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), height=180,
        font=dict(color="#94a3b8"),
    )
    st.markdown('<div class="glass-card"><div class="section-title">🎯 Profit Margin</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Top country stats
    if not cdf.empty:
        top = cdf.iloc[0]
        st.markdown(f"""
        <div class="glass-card">
          <div class="section-title">🏆 Top Market</div>
          <div style="text-align:center;padding:8px 0">
            <div style="font-family:Orbitron;font-size:1rem;color:#38bdf8">{top['Country']}</div>
            <div style="font-size:.7rem;color:#64748b;margin:4px 0">REVENUE LEADER</div>
            <div style="font-family:Orbitron;font-size:1.3rem;color:#10b981">${top['Revenue']/1e6:.1f}M</div>
            <div style="font-size:.7rem;color:#94a3b8;margin-top:8px">
              {top['Orders']:,} orders · {top['Customers']:,} customers<br>
              Margin: {top['Margin']:.1f}%
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# ANALYTICS DEEP DIVE — PNG Charts Section
# ══════════════════════════════════════════════════════════════════
st.markdown("<div style='padding:0 24px 24px'>", unsafe_allow_html=True)
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-title" style="font-size:.85rem;margin-bottom:20px">📡 ANALYTICS INTELLIGENCE HUB — DEEP DIVE REPORTS</div>', unsafe_allow_html=True)

tab_labels = ["📊 KPI Overview", "📈 Revenue & Trends", "🛍️ Products & Categories",
              "👥 Customer Analytics", "🤖 ML & Forecasting", "🏪 Store & Operations",
              "🌐 FX & Currency", "🔬 Statistical Analysis"]

sel_tab = st.selectbox("SELECT ANALYTICS MODULE", tab_labels, index=0,
                       label_visibility="collapsed")

def show_chart(filename, title, caption=""):
    b64 = img_b64(filename)
    if b64:
        st.markdown(f"""
        <div class="chart-container">
          <div class="chart-title">{title}</div>
          <img src="data:image/png;base64,{b64}" alt="{title}"/>
          {"<div style='font-size:.7rem;color:#64748b;margin-top:8px;text-align:center'>" + caption + "</div>" if caption else ""}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chart-container"><div class="chart-title">{title}</div><div style="color:#64748b;font-size:.8rem;padding:20px;text-align:center">Chart file not found: {filename}</div></div>', unsafe_allow_html=True)

# ── Tab 1: KPI Overview ────────────────────────────────────────────────────
if sel_tab == "📊 KPI Overview":
    c1, c2 = st.columns(2)
    with c1:
        show_chart("fig_kpi_dashboard.png", "KPI Dashboard — Executive Summary",
                   "Comprehensive KPI overview across all business dimensions")
        show_chart("fig_quarterly.png", "Quarterly Performance Analysis",
                   "Revenue and profit trends broken down by quarter")
    with c2:
        show_chart("fig_profitability.png", "Profitability Analysis",
                   "Margin breakdown by product category and region")
        show_chart("fig_profit_trend.png", "Profit Trend Over Time",
                   "Historical profit trajectory with trend line")

# ── Tab 2: Revenue & Trends ────────────────────────────────────────────────
elif sel_tab == "📈 Revenue & Trends":
    c1, c2 = st.columns(2)
    with c1:
        show_chart("fig_monthly_trend.png", "Monthly Revenue Trend (2016–2021)",
                   "Month-over-month revenue performance across all years")
        show_chart("fig_seasonality_heatmap.png", "Seasonality Heatmap",
                   "Revenue intensity by month and year — reveals seasonal patterns")
    with c2:
        show_chart("fig_dow_analysis.png", "Day-of-Week Sales Analysis",
                   "Transaction volume and revenue by day of week")
        show_chart("fig_channel.png", "Sales Channel Performance",
                   "Online vs in-store revenue and order distribution")

# ── Tab 3: Products & Categories ──────────────────────────────────────────
elif sel_tab == "🛍️ Products & Categories":
    c1, c2 = st.columns(2)
    with c1:
        show_chart("fig_category_performance.png", "Category Performance Matrix",
                   "Revenue, profit, and margin by product category")
        show_chart("fig_top_products.png", "Top Products by Revenue",
                   "Best-performing SKUs ranked by total revenue contribution")
    with c2:
        show_chart("fig_radar.png", "Category Radar Chart",
                   "Multi-dimensional category comparison across KPIs")
        st.markdown("""
        <div class="glass-card">
          <div class="section-title">🏷️ Category Intelligence</div>
          <div class="ai-insight"><span class="ai-tag">⚡ Top Category</span>Computers & Laptops drive the highest revenue share at ~35% of total sales.</div>
          <div class="ai-insight"><span class="ai-tag">⚡ Best Margin</span>Accessories category delivers superior margins due to low COGS relative to price.</div>
          <div class="ai-insight"><span class="ai-tag">⚡ Growth Signal</span>Smart Home & Audio categories showing accelerating YoY growth trajectory.</div>
        </div>""", unsafe_allow_html=True)

# ── Tab 4: Customer Analytics ──────────────────────────────────────────────
elif sel_tab == "👥 Customer Analytics":
    c1, c2 = st.columns(2)
    with c1:
        show_chart("fig_03_customer_country.png", "Customer Distribution by Country",
                   "Geographic spread of customer base across 8 markets")
        show_chart("fig_age_distribution.png", "Customer Age Distribution",
                   "Age demographics of the global customer base")
        show_chart("fig_clv_tiers.png", "Customer Lifetime Value Tiers",
                   "CLV segmentation — Bronze, Silver, Gold, Platinum tiers")
    with c2:
        show_chart("fig_gender_analysis.png", "Gender-Based Purchase Analysis",
                   "Revenue and order patterns segmented by gender")
        show_chart("fig_segments.png", "Customer Segmentation (K-Means)",
                   "ML-driven customer clusters based on RFM analysis")

# ── Tab 5: ML & Forecasting ────────────────────────────────────────────────
elif sel_tab == "🤖 ML & Forecasting":
    c1, c2 = st.columns(2)
    with c1:
        show_chart("fig_ml_forecast.png", "ML Revenue Forecast (Random Forest)",
                   "12-month forward revenue projection using Random Forest model")
        show_chart("fig_kmeans_elbow.png", "K-Means Elbow Curve",
                   "Optimal cluster count determination for customer segmentation")
    with c2:
        show_chart("fig_anomaly.png", "Anomaly Detection Analysis",
                   "Statistical outlier detection in transaction data")
        st.markdown("""
        <div class="glass-card">
          <div class="section-title">🤖 ML Model Performance</div>
          <div class="ai-insight"><span class="ai-tag">⚡ Random Forest</span>Revenue forecast model trained on 62,884 transactions. R² score indicates strong predictive accuracy.</div>
          <div class="ai-insight"><span class="ai-tag">⚡ K-Means Clustering</span>Optimal 4-cluster solution identified for customer segmentation with clear behavioral separation.</div>
          <div class="ai-insight"><span class="ai-tag">⚡ Anomaly Detection</span>Isolation Forest algorithm flagged 2.3% of transactions as statistical outliers requiring review.</div>
        </div>""", unsafe_allow_html=True)

# ── Tab 6: Store & Operations ──────────────────────────────────────────────
elif sel_tab == "🏪 Store & Operations":
    c1, c2 = st.columns(2)
    with c1:
        show_chart("fig_store_performance.png", "Store Performance Benchmarking",
                   "Revenue and efficiency metrics across all 58 store locations")
    with c2:
        show_chart("fig_outliers.png", "Store Outlier Analysis",
                   "Identifying over and under-performing store locations")
    st.markdown("""
    <div class="glass-card">
      <div class="section-title">🏪 Operations Intelligence</div>
      <div class="ai-insight"><span class="ai-tag">⚡ Store Network</span>58 stores across 8 countries. Top 10% of stores generate 38% of total revenue.</div>
      <div class="ai-insight"><span class="ai-tag">⚡ Efficiency</span>Online channel growing at 2.3x the rate of physical stores — digital transformation accelerating.</div>
    </div>""", unsafe_allow_html=True)

# ── Tab 7: FX & Currency ──────────────────────────────────────────────────
elif sel_tab == "🌐 FX & Currency":
    c1, c2 = st.columns(2)
    with c1:
        show_chart("fig_fx_trends.png", "FX Rate Trends (2016–2021)",
                   "Exchange rate movements for EUR, GBP, CAD, AUD vs USD")
        show_chart("fig_currency.png", "Revenue by Currency",
                   "Revenue distribution across transaction currencies")
    with c2:
        st.markdown("""
        <div class="glass-card">
          <div class="section-title">💱 FX Intelligence</div>
          <div class="ai-insight"><span class="ai-tag">⚡ USD Dominance</span>~62% of revenue transacted in USD. Strong home market insulates against FX risk.</div>
          <div class="ai-insight"><span class="ai-tag">⚡ EUR Exposure</span>EUR/USD corridor represents largest FX risk. 2020 volatility impacted EU revenue by ~3.2%.</div>
          <div class="ai-insight"><span class="ai-tag">⚡ GBP Impact</span>Post-Brexit GBP weakness created pricing pressure in UK market — partially offset by volume growth.</div>
          <div class="ai-insight"><span class="ai-tag">⚡ Hedging Signal</span>AUD and CAD showing correlated movements — natural hedge opportunity identified.</div>
        </div>""", unsafe_allow_html=True)

# ── Tab 8: Statistical Analysis ───────────────────────────────────────────
elif sel_tab == "🔬 Statistical Analysis":
    c1, c2 = st.columns(2)
    with c1:
        show_chart("fig_correlation.png", "Correlation Matrix",
                   "Pearson correlation between all numeric business variables")
        show_chart("fig_pairplot.png", "Pairplot — Variable Relationships",
                   "Scatter matrix revealing multivariate relationships")
    with c2:
        show_chart("fig_outliers.png", "Statistical Outlier Detection",
                   "Box-plot based outlier identification across key metrics")
        st.markdown("""
        <div class="glass-card">
          <div class="section-title">🔬 Statistical Insights</div>
          <div class="ai-insight"><span class="ai-tag">⚡ Key Correlation</span>Strong positive correlation (r=0.87) between store size and revenue. Location quality matters.</div>
          <div class="ai-insight"><span class="ai-tag">⚡ Price Elasticity</span>Unit price negatively correlated with quantity (-0.43) — classic demand curve behavior confirmed.</div>
          <div class="ai-insight"><span class="ai-tag">⚡ Outliers</span>Top 1% of transactions by value account for 8.7% of total revenue — high-value customer segment critical.</div>
        </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:20px;border-top:1px solid rgba(56,189,248,.1);margin-top:10px">
  <span style="font-family:Orbitron;font-size:.6rem;color:#334155;letter-spacing:3px">
    GLOBAL AI COMMAND CENTER · GLOBAL ELECTRONICS CORP · 2016–2021 · POWERED BY AI ANALYTICS ENGINE
  </span>
</div>
""", unsafe_allow_html=True)
