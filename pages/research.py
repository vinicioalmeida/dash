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
        <div class="card-title">Corporate Sustainable Bonds in Brazil: Market Reactions and Spillover Effects</div>
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
        <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6692059"
           target="_blank" style="font-size:0.8rem;">Download on SSRN ↗</a>
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
        <a href="https://papers.ssrn.com/abstract=6692178"
           target="_blank" style="font-size:0.8rem;">Download on SSRN ↗</a>
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
            <strong>Lemuel de Lemos Romão</strong> · 2023<br>
            <span style="color:#6b7280; font-size:0.85rem;">Fundos de investimento no Brasil</span><br><br>
            <strong>Irã Inácio Ribeiro</strong> · 2023<br>
            <span style="color:#6b7280; font-size:0.85rem;">Conexões políticas e precificação de ativos: evidências no mercado regulado brasileiro</span><br><br>
            <strong>Luís Othon Bastos</strong> · 2022<br>
            <span style="color:#6b7280; font-size:0.85rem;">O Frete de Retorno nas Ferrovias Regionais de Cargas Brasileiras</span><br><br>
            <strong>Robson Góes de Carvalho</strong> · 2021<br>
            <span style="color:#6b7280; font-size:0.85rem;">Família de índices de renda fixa: uma proposta metodológica</span><br><br>
            <strong>Giovanna Tonetto Segantini</strong> · 2019<br>
            <span style="color:#6b7280; font-size:0.85rem;">Two essays about mandatory dividend: what does the mandatory dividend have to inform to the market</span><br><br>
            <strong>Melquíades Pereira de Lima Júnior</strong> · 2014<br>
            <span style="color:#6b7280; font-size:0.85rem;">Desempenho de analistas sell-side no mercado de ações brasileiro</span>
        </div>
    </div>
    """, unsafe_allow_html=True)



    st.markdown(
        "<div class=\"section-label\">Master\'s Students &mdash; Advisor</div>",
        unsafe_allow_html=True
    )
    rows = [
        ("Thiago Wanderley Macedo Neves de Almeida", "2022",
         "Observa\u00e7\u00f5es e infer\u00eancias sobre o custo fixo na estrat\u00e9gia de put protetora"),
        ("Rodrigo Raposo da Fonseca", "2018",
         "A rela\u00e7\u00e3o entre otimismo e ondas de fus\u00e3o e aquisi\u00e7\u00e3o: evid\u00eancias do mercado brasileiro"),
        ("Heric Nero Lisboa dos Santos", "2018",
         "Compra de op\u00e7\u00f5es como alternativa para prote\u00e7\u00e3o de carteiras de a\u00e7\u00f5es"),
        ("Aletheia Januaria Zanow de Gouvea", "2016",
         "Insider Trading no Mercado de Capitais Brasileiro: O Crime Compensa?"),
        ("Patricia Ribeiro Romano", "2015",
         "Fus\u00f5es e aquisi\u00e7\u00f5es no Brasil: an\u00e1lise dos efeitos em mercado de capitais"),
        ("Tanite de Melo Silva", "2015",
         "Desempenho de fundos de investimento"),
        ("Ruan Rodrigo Ara\u00fajo da Costa", "2013",
         "A rela\u00e7\u00e3o entre desempenho e a forma legal das institui\u00e7\u00f5es de microcr\u00e9dito"),
        ("Luiz Carlos Santos J\u00fanior", "2012",
         "An\u00e1lise experimental do efeito diversifica\u00e7\u00e3o em carteiras de a\u00e7\u00f5es"),
        ("Lana Viviane Linhares da Costa Silva", "2012",
         "Teoria de carteiras aplicada \u00e0 an\u00e1lise de disposi\u00e7\u00e3o geogr\u00e1fica de usinas e\u00f3licas offshore"),
        ("Jo\u00e3o Paulo Costa de Medeiros", "2012",
         "Precifica\u00e7\u00e3o da energia e\u00f3lica offshore no Brasil"),
    ]
    body = "<br><br>".join(
        f"<strong>{n}</strong> \u00b7 {y}<br>"
        f'<span style="color:#6b7280; font-size:0.85rem;">{t}</span>'
        for n, y, t in rows
    )
    st.markdown(
        f'<div class="card"><div class="card-body">{body}</div></div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="footer">Vinicio Almeida \u00b7 DEPAD/PPGA \u00b7 UFRN</div>', unsafe_allow_html=True)
