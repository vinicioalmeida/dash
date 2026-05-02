import streamlit as st

st.markdown('<div class="page-title">Teaching</div>', unsafe_allow_html=True)

st.markdown("""
<div class="card" style="margin-bottom:1.5rem;">
    <div class="card-body">
        Course repositories are hosted at
        <a href="https://github.com/ufrnfinancas" target="_blank">github.com/ufrnfinancas ↗</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Graduate ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Graduate Level — PPGA/UFRN</div>', unsafe_allow_html=True)

courses_grad = [
    ("PPGA0054", "Python for Finance", "https://github.com/ufrnfinancas/PPGA0054-pythonfin",
     "Python programming applied to financial analysis, data wrangling, portfolio optimization, and quantitative modeling."),
    ("PPGA0158", "Capital Markets", "https://github.com/ufrnfinancas/PPGA0158-mercap",
     "Structure and functioning of Brazilian capital markets (B3), equity and fixed income instruments, and investment analysis."),
    ("PPGA0156", "Derivatives", "https://github.com/ufrnfinancas/PPGA0156-derivativos",
     "Options, futures, and structured products. Pricing models including binomial trees and Black-Scholes-Merton. Risk management strategies."),
]

for code, name, url, desc in courses_grad:
    st.markdown(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <div class="card-title">{name}</div>
            <span class="tag">{code}</span>
        </div>
        <div class="card-body" style="margin-top:0.3rem;">{desc}</div>
        <a href="{url}" target="_blank" style="font-size:0.8rem;">GitHub repository ↗</a>
    </div>
    """, unsafe_allow_html=True)

# ── Undergraduate ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Undergraduate Level — DEPAD/UFRN</div>', unsafe_allow_html=True)

courses_grad2 = [
    ("ADM0416", "Financial Management II", "https://github.com/ufrnfinancas/ADM0416-admfin2",
     "Corporate finance, capital budgeting, cost of capital, and valuation. Quantitative applications in Python."),
    ("ADM0515", "Portfolio Management", "https://github.com/ufrnfinancas/ADM0515-carteiras",
     "Modern portfolio theory, risk-return analysis, efficient frontier, CAPM, and performance measurement."),
]

for code, name, url, desc in courses_grad2:
    st.markdown(f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:baseline;">
            <div class="card-title">{name}</div>
            <span class="tag">{code}</span>
        </div>
        <div class="card-body" style="margin-top:0.3rem;">{desc}</div>
        <a href="{url}" target="_blank" style="font-size:0.8rem;">GitHub repository ↗</a>
    </div>
    """, unsafe_allow_html=True)

# ── EAD ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Distance Learning — EAD/UFRN</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <div class="card-title">Matemática para Administradores</div>
    <div class="card-body">
        Online course providing quantitative foundations for business administration students.
        Weekly guides covering financial mathematics and applied calculus.
    </div>
    <span class="tag">EAD</span>
    <span class="tag">Moodle</span>
    <span class="tag">Undergraduate</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="footer">Vinicio Almeida · DEPAD/PPGA · UFRN</div>', unsafe_allow_html=True)
