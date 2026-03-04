import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from data_fetch import get_stock_data
from monte_carlo import run_simulation
from risk_metrics import calculate_metrics
from valuation_engine import run_valuation
from financial_data import FUNDAMENTAL_DATA

st.set_page_config(page_title="Pro Stock Forecaster", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
STOCK_INFO = {
    "TATAMOTORS.NS": ("Tata Motors", "Rithin Reji"),
    "M&M.NS":        ("Mahindra & Mahindra", "Vinamra Gupta"),
    "OLECTRA.NS":    ("Olectra Greentech", "Aryan Jha"),
    "ATHERENERG.NS": ("Ather Energy", ""),
    "TSLA":    ("Tesla Inc.", ""),
    "P911.DE": ("Porsche AG", "Gautam Poturaju"),
    "F":       ("Ford Motor Co.", "Archana V"),
    "VOW3.DE": ("Volkswagen AG", "Sunidhi Datar"),
    "HYMTF":   ("Hyundai Motor", "Samarth Rao"),
    "APOLLOTYRE.NS": ("Apollo Tyres", "Anirudh Agarwal"),
    "MRF.NS":        ("MRF Ltd.", "Shrisai Hari"),
    "JKTYRE.NS":     ("JK Tyre & Industries", "Swayam Panigrahi"),
    "CEATLTD.NS":    ("CEAT Ltd.", "Harshini Venkat"),
    "SBIN.NS":      ("State Bank of India", "Anoushka Gadhwal"),
    "HDFCBANK.NS":  ("HDFC Bank", "Ryan Kidangan"),
    "ICICIBANK.NS": ("ICICI Bank", "Himangshi Bose"),
    "AXISBANK.NS":  ("Axis Bank", "Bismaya Nayak"),
    "LAURUSLABS.NS": ("Laurus Labs", "Satvik Sharma"),
    "AUROPHARMA.NS": ("Aurobindo Pharma", "Arya Mukharjee"),
    "SUNPHARMA.NS":  ("Sun Pharma", "Yogesh Bolkotagi"),
    "DIVISLAB.NS":   ("Divi's Laboratories", "Bhavansh Madan"),
    "ITC.NS":       ("ITC Ltd.", "Gajanan Kudva / Srutayus Das"),
    "CHALET.NS":    ("Chalet Hotels", "Shreya Joshi"),
    "MHRIL.NS":     ("Mahindra Holidays", "Gowri Shetty"),
    "INDHOTEL.NS":  ("Indian Hotels Co.", "Aarohi Jain"),
    "HUL.NS":       ("Hindustan Unilever", "Suhina Sarkar"),
    "NESTLEIND.NS": ("Nestlé India", "Saaraansh Razdan"),
    "SHREECEM.NS":   ("Shree Cement", "Anjor Singh"),
    "ULTRACEMCO.NS": ("UltraTech Cement", "Rahul Gowda"),
    "DALBHARAT.NS":  ("Dalmia Bharat", "Kushagra Shukla"),
    "RAMCOCEM.NS":   ("Ramco Cements", "Grace Rebecca David"),
    "ABSLAMC.NS":    ("Aditya Birla Sun Life AMC", "Pallewar Pranav"),
    "HDFCAMC.NS":    ("HDFC AMC", "Rittika Saraswat"),
    "NAM-INDIA.NS":  ("Nippon Life India AMC", "Sam Phillips"),
    "UTIAMC.NS":     ("UTI AMC", "Abhinav Singh"),
    "NVDA":  ("NVIDIA Corp.", "Sijal Verma"),
    "MSFT":  ("Microsoft Corp.", "Gurleen Kaur"),
    "GOOGL": ("Alphabet Inc.", "Anugraha AB"),
    "META":  ("Meta Platforms", "Senjuti Pal"),
    "IBM":   ("IBM Corp.", "Biba Pattnaik"),
    "ASML":  ("ASML Holding", "Adaa Gujral"),
    "INTC":  ("Intel Corp.", "Aditi Ranjan"),
    "QCOM":  ("Qualcomm Inc.", "Arpit Sharma"),
    "CRM":   ("Salesforce Inc.", "Rishit Hotchandani"),
    "PLTR":  ("Palantir Technologies", "Krrish Bahuguna"),
    "CRWD":  ("CrowdStrike Holdings", "Ashi Beniwal"),
    "WBD":  ("Warner Bros. Discovery", "Dhairya Vanker"),
    "NFLX": ("Netflix Inc.", "Hiya Phatnani"),
    "DIS":  ("Walt Disney Co.", "Siya Sharma"),
    "PARA": ("Paramount Global", "Tanvi Gujarathi"),
    "PG":   ("Procter & Gamble", "Nayan Kanchan"),
    "WMT":  ("Walmart Inc.", ""),
    "LMT": ("Lockheed Martin", "Siddhant Mehta"),
    "GD":  ("General Dynamics", "Shlok Pratap Singh"),
    "NOC": ("Northrop Grumman", "Harshdeep Roshan"),
    "RTX": ("RTX Corporation", "Prandeep Poddar"),
}


def _is_indian(t):
    return t.endswith(".NS") or t.endswith(".BO")


def _cur(t):
    return "₹" if _is_indian(t) else "$"


def _fmt(v, t):
    return f"{_cur(t)}{v:,.2f}"


GLOBAL_STOCKS = {
    "Auto (India)":              ["TATAMOTORS.NS", "M&M.NS", "OLECTRA.NS", "ATHERENERG.NS"],
    "Auto (Global)":             ["TSLA", "P911.DE", "F", "VOW3.DE", "HYMTF"],
    "Tyres (India)":             ["APOLLOTYRE.NS", "MRF.NS", "JKTYRE.NS", "CEATLTD.NS"],
    "Banking (India)":           ["SBIN.NS", "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS"],
    "Pharma (India)":            ["LAURUSLABS.NS", "AUROPHARMA.NS", "SUNPHARMA.NS", "DIVISLAB.NS"],
    "Consumer & Hotels (India)": ["ITC.NS", "CHALET.NS", "MHRIL.NS", "INDHOTEL.NS", "HUL.NS", "NESTLEIND.NS"],
    "Cement (India)":            ["SHREECEM.NS", "ULTRACEMCO.NS", "DALBHARAT.NS", "RAMCOCEM.NS"],
    "AMC / Finance (India)":     ["ABSLAMC.NS", "HDFCAMC.NS", "NAM-INDIA.NS", "UTIAMC.NS"],
    "Tech (US/Global)":          ["NVDA", "MSFT", "GOOGL", "META", "IBM", "ASML", "INTC", "QCOM", "CRM", "PLTR", "CRWD"],
    "Media & Consumer (US)":     ["WBD", "NFLX", "DIS", "PARA", "PG", "WMT"],
    "Defense (US)":              ["LMT", "GD", "NOC", "RTX"],
}


def _display_name(t):
    info = STOCK_INFO.get(t)
    if info:
        c, p = info
        return f"{c} ({p})" if p else c
    return t


# ═════════════════════════════════════════════════════════════════════════════
st.title("🏛️ Institutional Equity Lab & Forecasting")
st.markdown("*Damodaran DCF Valuation (with full calculation trail) + Monte Carlo Risk Simulation*")
st.markdown("---")

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("1. Market Selection")
category = st.sidebar.selectbox("Filter by Sector", list(GLOBAL_STOCKS.keys()))
ticker_list = GLOBAL_STOCKS[category]
display_list = [_display_name(t) for t in ticker_list]
selected_display = st.sidebar.selectbox("Select Company", display_list)
selected_ticker = ticker_list[display_list.index(selected_display)]

custom_ticker = st.sidebar.text_input("OR Type Custom (e.g. RELIANCE.NS)")
ticker = custom_ticker.strip() if custom_ticker.strip() else selected_ticker
cur = _cur(ticker)

st.sidebar.header("2. Simulation Parameters")
sims = st.sidebar.slider("Number of Simulations", 5000, 50000, 10000)
years = st.sidebar.slider("Investment Horizon (Years)", 0.5, 10.0, 1.0)
crash_scenario = st.sidebar.slider("Simulate Market Stress (%)", 0, 50, 0)

try:
    info = STOCK_INFO.get(ticker, (ticker, ""))
    co_name, person = info
    person_badge = f"&nbsp;&nbsp;👤 <i>{person}</i>" if person else ""
    st.markdown(
        f"<h2>{co_name} &nbsp;<code>{ticker}</code>{person_badge}</h2>",
        unsafe_allow_html=True,
    )

    has_fundamentals = ticker in FUNDAMENTAL_DATA
    val = None
    intrinsic = 0

    # ════════════════════════════════════════════════════════════════════════
    # SECTION A: DAMODARAN DCF VALUATION
    # ════════════════════════════════════════════════════════════════════════
    if has_fundamentals:
        with st.spinner("Running Damodaran DCF Valuation…"):
            val = run_valuation(ticker)

        mc = val["model_selection"]
        vd = val["valuation_detail"]
        fd = val["fundamentals"]
        comp = val["computed"]
        intrinsic = val["intrinsic_value_per_share"]
        unit = fd["unit"]

        # ────────────────────────────────────────────────────────────────
        # STEP 1: MODEL SELECTION — Full Q&A (replicates model1.xls)
        # ────────────────────────────────────────────────────────────────
        st.subheader("📐 STEP 1: Choosing the Right Valuation Model")
        st.caption("Replicates Damodaran's model1.xls — every question answered with our data")

        # Show the Q&A inputs as a formatted table
        qa = mc["qa_inputs"]
        with st.expander("📝 Inputs to the Model (Full Q&A from model1.xls)", expanded=True):
            current_section = ""
            for item in qa:
                section = item.get("section", "")
                if section and section != current_section:
                    st.markdown(f"**━━━ {section} ━━━**")
                    current_section = section

                q = item["question"]
                a = item["answer"]

                if "formula" in item:
                    st.markdown(f"**{q}**")
                    st.code(item["formula"], language="text")
                    st.markdown(f"**= {a}**")
                else:
                    st.markdown(f"**{q}** → `{a}`")

                if "note" in item:
                    st.caption(f"ℹ️ {item['note']}")

        # Show the decision trail
        with st.expander("🧠 Decision Trail — How We Arrived at the Model", expanded=True):
            for i, step in enumerate(mc["decision_trail"], 1):
                st.markdown(f"{i}. {step}")

        # Final model selection box
        st.markdown(
            f"""
            <div style="background:#0d1117;padding:20px;border-radius:12px;
                        border:2px solid #58a6ff;margin:15px 0;">
                <h3 style="color:#58a6ff;margin:0;">OUTPUT FROM MODEL SELECTOR</h3>
                <table style="color:white;width:100%;margin-top:12px;font-size:16px;">
                    <tr><td style="padding:6px;color:#8b949e;">Type of Model:</td>
                        <td style="padding:6px;"><b>{mc['model_type']}</b></td></tr>
                    <tr><td style="padding:6px;color:#8b949e;">Earnings Level:</td>
                        <td style="padding:6px;"><b>{mc['earnings_level']}</b></td></tr>
                    <tr><td style="padding:6px;color:#8b949e;">Cashflows to Discount:</td>
                        <td style="padding:6px;"><b>{mc['cashflow_type']}</b></td></tr>
                    <tr><td style="padding:6px;color:#8b949e;">Growth Period:</td>
                        <td style="padding:6px;"><b>{mc['growth_period']}</b></td></tr>
                    <tr><td style="padding:6px;color:#8b949e;">Growth Pattern:</td>
                        <td style="padding:6px;"><b>{mc['growth_pattern']}</b></td></tr>
                    <tr><td style="padding:6px;color:#8b949e;">Selected Model:</td>
                        <td style="padding:6px;"><b>{mc['model_description']}</b>
                            &nbsp;<code>{mc['model_code']}.xls</code></td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ────────────────────────────────────────────────────────────────
        # STEP 2: DCF INPUTS (Annual Report Data)
        # ────────────────────────────────────────────────────────────────
        st.subheader("📊 STEP 2: Annual Report Data (DCF Inputs)")

        in1, in2, in3 = st.columns(3)
        with in1:
            st.markdown("**Income Statement**")
            st.write(f"Revenue: {cur}{fd['revenue']:,.0f} {unit}")
            st.write(f"EBIT: {cur}{fd['ebit']:,.0f} {unit}")
            st.write(f"Net Income: {cur}{fd['net_income']:,.0f} {unit}")
            st.write(f"EPS: {cur}{comp['EPS']:,.2f}")
            st.write(f"Tax Rate: {fd['tax_rate']:.0%}")

        with in2:
            st.markdown("**Cash Flow Statement**")
            st.write(f"Depreciation: {cur}{fd['depreciation']:,.0f} {unit}")
            st.write(f"CapEx: {cur}{fd['capex']:,.0f} {unit}")
            st.write(f"Δ Working Capital: {cur}{fd['delta_wc']:,.0f} {unit}")
            st.write(f"Dividends: {cur}{fd['dividends_total']:,.0f} {unit}")
            st.write(f"DPS: {cur}{comp['DPS']:,.2f}")
            st.write(f"**FCFE Total: {cur}{comp['FCFE_total']:,.0f} {unit}**")
            st.write(f"**FCFF Total: {cur}{comp['FCFF_total']:,.0f} {unit}**")

        with in3:
            st.markdown("**Balance Sheet & Rates**")
            st.write(f"Total Debt: {cur}{fd['total_debt']:,.0f} {unit}")
            st.write(f"Cash: {cur}{fd['cash']:,.0f} {unit}")
            st.write(f"Shares: {fd['shares_outstanding']:,.2f} {unit}")
            st.write(f"Debt Ratio: {fd['debt_ratio']:.1%}")
            st.write(f"Beta: {fd['beta']:.2f}")
            st.write(f"Risk-Free Rate: {fd['risk_free_rate']:.1%}")
            st.write(f"Equity Risk Premium: {fd['erp']:.1%}")
            st.write(f"Cost of Equity (Ke): {fd['cost_of_equity']:.1%}")
            st.write(f"WACC: {fd['wacc']:.1%}")

        st.markdown("---")

        # ────────────────────────────────────────────────────────────────
        # STEP 3: DCF OUTPUT — Year-by-year table + Summary
        # ────────────────────────────────────────────────────────────────
        st.subheader(f"💎 STEP 3: {vd.get('model', 'DCF')} — Year-by-Year Calculation")

        # Year-by-year table
        if "year_by_year" in vd and vd["year_by_year"]:
            yby = vd["year_by_year"]
            df_yby = pd.DataFrame(yby)

            # Format numeric columns
            for col in df_yby.columns:
                if col in ("Year", "Phase", "Growth", "Growth Rate",
                           "Expected Growth", "Cost of Equity", "WACC"):
                    continue
                if df_yby[col].dtype in [np.float64, np.int64, float, int]:
                    df_yby[col] = df_yby[col].apply(
                        lambda x: f"{cur}{x:,.2f}" if abs(x) >= 1 else f"{x:.6f}"
                    )

            st.dataframe(df_yby, use_container_width=True, hide_index=True)

        # Formula display
        if "formula" in vd:
            st.code(vd["formula"], language="text")

        # Summary table
        if "summary" in vd:
            st.markdown("**Valuation Summary:**")
            summary = vd["summary"]
            sum_rows = []
            for k, v in summary.items():
                if isinstance(v, float):
                    if abs(v) < 1 and v != 0:
                        formatted = f"{v:.2%}"
                    else:
                        formatted = f"{cur}{v:,.2f}"
                elif isinstance(v, str):
                    formatted = v
                else:
                    formatted = f"{v:,}" if isinstance(v, int) else str(v)
                sum_rows.append({"Item": k, "Value": formatted})

            st.table(pd.DataFrame(sum_rows))

        # Big intrinsic value display
        if intrinsic > 0:
            st.markdown(
                f"""
                <div style="background:#0d1117;padding:25px;border-radius:12px;
                            border:3px solid #00d4ff;text-align:center;margin:20px 0;">
                    <h1 style="color:#00d4ff;margin:0;">
                        Intrinsic Value = {_fmt(intrinsic, ticker)} per share
                    </h1>
                    <p style="color:#8b949e;margin-top:8px;font-size:16px;">
                        Model: {mc['model_description']} ({mc['model_code']}.xls)
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ Intrinsic value ≤ 0. Inputs may need adjustment.")

        st.markdown("---")
    else:
        st.info("ℹ️ No fundamental data available for DCF valuation. Showing Monte Carlo only.")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION B: MONTE CARLO
    # ════════════════════════════════════════════════════════════════════════
    st.subheader("📈 STEP 4: Monte Carlo Risk Simulation")

    with st.spinner(f"Fetching market price for {ticker}…"):
        s0, auto_mu, auto_sigma, source_name = get_stock_data(ticker)

    st.sidebar.success(f"✅ Price: {source_name}")

    adjusted_mu = auto_mu - (crash_scenario / 100)

    with st.spinner("Running Monte Carlo…"):
        paths, low_band, high_band = run_simulation(
            s0, adjusted_mu, auto_sigma, years, n_sims=sims
        )
    final_prices = paths[-1]
    metrics = calculate_metrics(final_prices, s0, adjusted_mu, auto_sigma)

    # ── Verdict banner ──────────────────────────────────────────────────
    if val and intrinsic > 0:
        mos = (intrinsic - s0) / s0
        if mos > 0.20:
            vc, vt = "#1b8a2a", "🟢 UNDERVALUED — BUY"
        elif mos > -0.10:
            vc, vt = "#c47f17", "🟡 FAIRLY VALUED — HOLD"
        else:
            vc, vt = "#b52a2a", "🔴 OVERVALUED — AVOID"

        st.markdown(
            f"""
            <div style="background:{vc};padding:25px;border-radius:12px;text-align:center;margin-bottom:20px;">
                <h1 style="color:white;margin:0;">{vt}</h1>
                <h3 style="color:white;margin-top:10px;">
                    Market: {_fmt(s0, ticker)} &nbsp;|&nbsp;
                    DCF: {_fmt(intrinsic, ticker)} &nbsp;|&nbsp;
                    Margin of Safety: {mos:.1%} &nbsp;|&nbsp;
                    MC Target ({years}yr): {_fmt(metrics['Expected Price'], ticker)}
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if "BUY" in metrics["Signal"]:
            sc = "#1b8a2a"
        elif "HOLD" in metrics["Signal"]:
            sc = "#c47f17"
        else:
            sc = "#b52a2a"
        st.markdown(
            f"""
            <div style="background:{sc};padding:25px;border-radius:12px;text-align:center;margin-bottom:20px;">
                <h1 style="color:white;margin:0;">SIGNAL: {metrics['Signal']}</h1>
                <h3 style="color:white;margin-top:10px;">
                    MC Target ({years}yr): {_fmt(metrics['Expected Price'], ticker)}
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Key metrics row ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", _fmt(s0, ticker))
    if val and intrinsic > 0:
        c2.metric("DCF Intrinsic Value", _fmt(intrinsic, ticker), f"{((intrinsic/s0)-1):.1%}")
    else:
        c2.metric("MC Expected", _fmt(metrics["Expected Price"], ticker),
                  f"{((metrics['Expected Price']/s0)-1):.1%}")
    c3.metric("VaR 95%", f"{metrics['VaR 95% (Rel)']:.1%}")
    c4.metric("Prob. of Profit", f"{metrics['Prob. of Profit']:.1f}%")

    st.markdown("---")

    # ── Detailed risk metrics ───────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown("**📈 Price Targets**")
        st.write(f"Expected: {_fmt(metrics['Expected Price'], ticker)}")
        st.write(f"Median: {_fmt(metrics['Median Price'], ticker)}")
        st.write(f"Best: {_fmt(metrics['Best Case Price'], ticker)}")
        st.write(f"Worst: {_fmt(metrics['Worst Case Price'], ticker)}")
        st.write(f"90th: {_fmt(metrics['90th Percentile Price'], ticker)}")
        st.write(f"10th: {_fmt(metrics['10th Percentile Price'], ticker)}")
    with d2:
        st.markdown("**⚠️ Risk**")
        st.write(f"VaR 95%: {metrics['VaR 95% (Rel)']:.2%}")
        st.write(f"CVaR 95%: {metrics['CVaR 95%']:.2%}")
        st.write(f"VaR 99%: {metrics['VaR 99% (Rel)']:.2%}")
        st.write(f"CVaR 99%: {metrics['CVaR 99%']:.2%}")
        st.write(f"Max Drawdown: {metrics['Max Drawdown']:.1f}%")
        st.write(f"Volatility: {metrics['Volatility (Annual)']:.1f}%")
    with d3:
        st.markdown("**📊 Probability**")
        st.write(f"Profit: {metrics['Prob. of Profit']:.1f}%")
        st.write(f">10% Gain: {metrics['Prob. of >10% Gain']:.1f}%")
        st.write(f">25% Gain: {metrics['Prob. of >25% Gain']:.1f}%")
        st.write(f">10% Loss: {metrics['Prob. of >10% Loss']:.1f}%")
        st.write(f"Avg Up: +{metrics['Avg Upside']:.1f}%")
        st.write(f"Avg Down: {metrics['Avg Downside']:.1f}%")
    with d4:
        st.markdown("**🏆 Ratios**")
        st.write(f"Sharpe: {metrics['Sharpe Ratio']:.2f}")
        st.write(f"Sortino: {metrics['Sortino Ratio']:.2f}")
        st.write(f"Risk-Reward: {metrics['Risk-Reward Ratio']:.2f}")
        st.write(f"Exp. Return: {metrics['Expected Return']:.1f}%")
        st.write(f"Max Upside: +{metrics['Max Upside']:.1f}%")

    st.markdown("---")

    # ── Charts ──────────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        x = np.arange(len(low_band))
        fig.add_trace(go.Scatter(x=x, y=high_band, fill=None, mode="lines",
                                  line_color="rgba(0,255,0,0.1)", name="Top 5%"))
        fig.add_trace(go.Scatter(x=x, y=low_band, fill="tonexty", mode="lines",
                                  line_color="rgba(255,0,0,0.1)", name="Bottom 5%"))
        fig.add_trace(go.Scatter(y=np.mean(paths, axis=1), mode="lines",
                                  line=dict(color="gold", width=3), name="Expected"))
        if val and intrinsic > 0:
            fig.add_hline(y=intrinsic, line_dash="dot", line_color="cyan",
                          annotation_text=f"DCF {_fmt(intrinsic, ticker)}")
        fig.update_layout(title="Monte Carlo Confidence Bands", template="plotly_dark",
                          xaxis_title="Days", yaxis_title=f"Price ({cur})")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = px.histogram(final_prices, nbins=60, title="Terminal Price Distribution",
                            template="plotly_dark", color_discrete_sequence=["#00CC96"])
        fig2.add_vline(x=s0, line_dash="dash", line_color="yellow",
                       annotation_text=f"Current {_fmt(s0, ticker)}")
        if val and intrinsic > 0:
            fig2.add_vline(x=intrinsic, line_dash="dot", line_color="cyan",
                           annotation_text=f"DCF {_fmt(intrinsic, ticker)}")
        fig2.update_layout(xaxis_title=f"Price ({cur})", yaxis_title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Benchmark ───────────────────────────────────────────────────────
    st.subheader("📊 Performance vs. Benchmarks")
    bn, br = ("Nifty 50 (12%)", 1.12) if _is_indian(ticker) else ("S&P 500 (10%)", 1.10)
    mg = s0 * (br ** years)
    rows = []
    if val and intrinsic > 0:
        rows.append([f"{co_name} — DCF", _fmt(intrinsic, ticker), f"{((intrinsic/s0)-1):.1%}"])
    rows.append([f"{co_name} — MC", _fmt(metrics["Expected Price"], ticker),
                 f"{((metrics['Expected Price']/s0)-1):.1%}"])
    rows.append([bn, _fmt(mg, ticker), f"{((mg/s0)-1):.1%}"])
    st.table(pd.DataFrame(rows, columns=["Scenario", "Value", "Return"]))

    st.caption(
        f"Price Source: {source_name} • "
        f"DCF: {mc['model_code'] if val else 'N/A'} • "
        f"Sims: {sims:,} • Horizon: {years}yr • Stress: {crash_scenario}%"
    )

except ValueError as e:
    st.error(f"Data Error: {e}")
except Exception as e:
    st.error(f"Error for {ticker}: {type(e).__name__}: {e}")
    st.exception(e)