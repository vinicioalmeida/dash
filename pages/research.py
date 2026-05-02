import streamlit as st

st.markdown('<div class="page-title">Research</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Working Papers", "Doctoral Supervision"])

with tab1:
    st.markdown("""
    <div class="card" style="margin-bottom:1rem;">
        <div class="card-body">
            For published articles, see
            <a href="https://scholar.google.com.br/citations?user=S-KEieUAAAAJ&hl=pt-BR"
               target="_blank">Google Scholar ↗</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Working Papers</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Corporate Sustainable Bonds in Brazil</div>
        <div class="card-meta">with Camila Farias and Israel Felipe</div>
        <div class="card-body">
            Event study on 62 green bond issuances in the Brazilian market (2015–2024).
            Results show negative cumulative abnormal returns of −1.71% over 21-day windows,
            contrasting with findings from developed markets.
            Winner of the <strong>ANBIMA Capital Markets Prize</strong>.
        </div>
        <span class="tag">Green Bonds</span>
        <span class="tag">Event Study</span>
        <span class="tag">Sustainable Finance</span>
    </div>

    <div class="card">
        <div class="card-title">Volatility Scaling in Multi-Asset Portfolios: Evidence from a Systematic Risk-Targeting Strategy</div>
        <div class="card-meta">with Camila Farias</div>
        <div class="card-body">
            Examines the performance of volatility-scaled allocation strategies across
            multi-asset portfolios, using a systematic risk-targeting approach applied
            to Brazilian and international markets.
        </div>
        <span class="tag">Volatility Scaling</span>
        <span class="tag">Portfolio Management</span>
        <span class="tag">Risk Targeting</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Work in Progress</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">Straddle Profitability in Brazil: Volatility Risk Premium and Trading Windows in Emerging Market Options</div>
        <div class="card-meta">Work in progress</div>
        <div class="card-body">
            Investigates the profitability of straddle strategies in the Brazilian options market,
            focusing on the volatility risk premium and the identification of optimal trading
            windows in an emerging market context.
        </div>
        <span class="tag">Options</span>
        <span class="tag">Volatility Risk Premium</span>
        <span class="tag">Straddle</span>
        <span class="tag">Brazil</span>
    </div>

    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-label">Doctoral Students — Advisor</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="card-body">
            <strong>Lemuel Lemos Romão</strong> · 2023<br>
            <strong>Irã Inácio Ribeiro</strong> · 2023<br>
            <strong>Robson Goes de Carvalho</strong> · 2021<br>
            <strong>Giovanna Tonetto Segantini</strong> · 2019<br>
            <strong>Melquíades Pereira de Lima Júnior</strong> · 2014
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">Vinicio Almeida · DEPAD/PPGA · UFRN</div>', unsafe_allow_html=True)
