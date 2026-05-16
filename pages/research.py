import streamlit as st

st.markdown('<div class="page-title">Research</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Publications", "Working Papers", "Supervision"])

with tab1:
    articles = [
        ("2026",
         "Governan\u00e7a e sustentabilidade em hospitais universit\u00e1rios: o caso da MEJC/EBSERH",
         "Farias, Almeida, Costa & Rodrigues Filho",
         "ReGeo",
         None),
        ("2025",
         "Observations and Inferences About the Fixed Cost in Protective Put Strategy",
         "Almeida & Almeida",
         "International Journal of Finance and Banking Studies, v.14, p.19--33",
         "https://www.ssbfnet.com/ojs/index.php/ijfbs/article/view/3848"),
        ("2025",
         "1+1>2: Integrating Analytical Techniques in the Age of AI",
         "Limongi, Brei, Almeida & Francisco",
         "BAR \u2014 Brazilian Administration Review, v.22, p.1--7",
         "https://www.scielo.br/j/bar/a/wgFSXS7xCPXZ75NL3v53vnf/?lang=en"),
        ("2024",
         "ESG Factors, Returns and Volatility: a Tale from Brazilian Market Data",
         "Farias & Almeida",
         "RGSA \u2014 Revista de Gest\u00e3o Social e Ambiental, v.18, p.1--15",
         "https://rgsa.openaccesspublications.org/rgsa/article/view/10238"),
        ("2023",
         "Rela\u00e7\u00e3o entre otimismo e ondas de fus\u00e3o e aquisi\u00e7\u00e3o: Evid\u00eancias do mercado brasileiro",
         "Fonseca & Almeida",
         "RAE \u2014 Revista de Administra\u00e7\u00e3o de Empresas, v.63, p.1--18",
         "https://www.scielo.br/j/rae/a/dCdfznXSmSTDGWCSjKkTxHP/?lang=pt"),
        ("2019",
         "Brazil offshore wind resources and atmospheric surface layer stability",
         "Pimenta, Silva, Assireu, Almeida & Saavedra",
         "Energies, v.12, p.4195",
         "https://www.mdpi.com/1996-1073/12/21/4195"),
        ("2018",
         "Bayesian Bid Updating in Experimental IPO Pricing Methods",
         "Almeida & Leal",
         "Revista de Finan\u00e7as Aplicadas, v.9, p.1--23",
         "http://www.financasaplicadas.fia.com.br/index.php/financasaplicadas/article/view/293"),
        ("2017",
         "Time span does matter for offshore wind plant allocation with modern portfolio theory",
         "Silva, Almeida, Pimenta & Segantini",
         "International Journal of Energy Economics and Policy, v.7, p.188--193",
         "https://dergipark.org.tr/en/download/article-file/361779"),
        ("2015",
         "Os analistas sell-side fazem boas previs\u00f5es de pre\u00e7os-alvo no Brasil?",
         "Lima J\u00fanior & Almeida",
         "Revista Brasileira de Finan\u00e7as, v.13, p.365--393",
         "https://periodicos.fgv.br/rbfin/article/view/35208"),
        ("2015",
         "An\u00e1lise dos efeitos em mercado de capitais decorrentes de fus\u00f5es: O caso BRF",
         "Romano & Almeida",
         "RAC \u2014 Revista de Administra\u00e7\u00e3o Contempor\u00e2nea, v.19, p.606--625",
         "https://www.scielo.br/j/rac/a/dq6gb9nhPLQWMKGHW74Jmvk/?lang=pt"),
        ("2015",
         "Brazilian initial public offerings, underwriters, and premium corporate governance segments listing",
         "Almeida & Leal",
         "Corporate Ownership & Control, v.13, p.1053--1061",
         "https://virtusinterpress.org/BRAZILIAN-INITIAL-PUBLIC-OFFERINGS.html"),
        ("2014",
         "A joint experimental analysis of investor behavior in IPO pricing methods",
         "Almeida & Leal",
         "RAE \u2014 Revista de Administra\u00e7\u00e3o de Empresas, v.55, p.14--25",
         "https://www.scielo.br/j/rae/a/tq9qkcw7QbncF3PdxKPb9qC/?lang=en&format=html"),
        ("2013",
         "Produ\u00e7\u00e3o cient\u00edfica brasileira em finan\u00e7as no per\u00edodo 2000--2010",
         "Leal, Almeida & Bortolon",
         "RAE \u2014 Revista de Administra\u00e7\u00e3o de Empresas, v.53, p.46--55",
         "https://www.scielo.br/j/rae/a/LJpMVNLzknYsMkCZL4Bm8qN/?format=html&lang=pt"),
        ("2013",
         "Evid\u00eancias na proje\u00e7\u00e3o do Value-at-Risk em pre\u00e7os de camar\u00e3o no Brasil via modelagem ARIMA com erros GARCH",
         "Felipe, Mol & Almeida",
         "Custos e @groneg\u00f3cio Online, v.9, p.49--78",
         "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2580508"),
        ("2012",
         "Are betas best? The correlation structure on Brazilian equity market",
         "Almeida",
         "Journal of International Finance and Economics, v.12, p.27--33",
         "https://jife-journal.org/JIFE-Journal/Documents/Abstracts/JIFE-12-1_Abstracts.pdf"),
        ("2012",
         "Large pension funds and the corporate governance practices of Brazilian companies",
         "Oliveira, Leal & Almeida",
         "Corporate Ownership & Control, v.9, p.76--84",
         "https://virtusinterpress.org/LARGE-PENSION-FUNDS-AND-THE.html"),
        ("2012",
         "Cointegrating to trade pairs in Brazilian stock market",
         "Almeida, Mol & Nascimento",
         "Journal of Academy of Business and Economics, v.12, p.12--22",
         "https://jabe-journal.org/JABE-Journal/Documents/Abstracts/JABE-12-1_Abstracts.pdf"),
        ("2012",
         "Desvendando o Book Building em ofertas de a\u00e7\u00f5es",
         "Romano & Almeida",
         "GVcasos, v.2",
         "https://periodicos.fgv.br/gvcasos/article/download/3866/2676"),
        ("2012",
         "Recupera\u00e7\u00e3o judicial de uma empresa a\u00e9rea em crise econ\u00f4mico-financeira",
         "Almeida & Romano",
         "GVcasos, v.2",
         "https://periodicos.fgv.br/gvcasos/article/download/3645/4660"),
        ("2012",
         "Application of ARIMA models in soybean series of prices in the North of Paran\u00e1",
         "Felipe, Mol, Almeida & Brito",
         "Custos e @groneg\u00f3cio Online, v.8, p.78--91",
         "https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=2580537"),
        ("2011",
         "Um \u00edndice de m\u00ednima vari\u00e2ncia de a\u00e7\u00f5es brasileiras",
         "Thome Neto, Leal & Almeida",
         "Economia Aplicada, v.15, p.615--633",
         "https://www.scielo.br/j/ecoa/a/3SfP56RC9YvjpKdHwSfpLsw/?lang=pt"),
        ("2011",
         "Underwriter reputation in Brazilian IPOs",
         "Almeida",
         "Latin American Business Review, v.12, p.255--280",
         "https://www.tandfonline.com/doi/full/10.1080/10978526.2011.633309"),
        ("2011",
         "Corporate governance in the context of the recovery of distressed firms",
         "Almeida & Romano",
         "Corporate Ownership & Control, v.9, p.621--627",
         "https://pdfs.semanticscholar.org/c156/d5a8b9837c63909965137bf0cfece0ed9875.pdf"),
        ("2010",
         "Shareholder base management in companies in the New Market listing segment of the Bovespa stock market",
         "Cals, Colares & Almeida",
         "Corporate Ownership & Control, v.8, p.226--236",
         "https://virtusinterpress.org/SHAREHOLDER-BASE-MANAGEMENT-IN.html"),
    ]

    for year, title, authors, journal, url in articles:
        link_html = (
            f'<a href="{url}" target="_blank" style="display:inline-block; margin-top:0.4rem; '            f'font-size:0.78rem; font-weight:600; color:#ffffff; background:#1a3550; '            f'padding:2px 10px; border-radius:3px; text-decoration:none;">Link &#8599;</a>'
            if url else
            '<span style="display:inline-block; margin-top:0.4rem; font-size:0.78rem; '            'font-weight:600; color:#b8973a; border:1px solid #b8973a; '            'padding:2px 10px; border-radius:3px;">Forthcoming</span>'
        )
        st.markdown(f"""
        <div class="card" style="padding:0.85rem 1.1rem;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;">
                <div style="flex:1;">
                    <div style="font-family:'Lora',serif; font-size:0.95rem; font-weight:600;
                                color:#0d1b2a; line-height:1.45; margin-bottom:0.25rem;">{title}</div>
                    <div style="font-size:0.8rem; color:#6b7280; margin-bottom:0.15rem;">{authors}</div>
                    <div style="font-size:0.78rem; color:#9ca3af; font-style:italic;">{journal if journal else ""}</div>
                    {link_html}
                </div>
                <div style="white-space:nowrap; font-size:0.8rem; font-weight:600;
                            color:#b8973a; padding-top:2px;">{year}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="section-label">Working Papers</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="card-title">Corporate Sustainable Bonds in Brazil: Market Reactions and Spillover Effects</div>
        <div class="card-meta">with Camila Farias and Israel Felipe</div>
        <div class="card-body">
            Event study on 62 green bond issuances in the Brazilian market (2015\u20132024).
            Results show negative cumulative abnormal returns of \u22121.71% over 21-day windows,
            contrasting with findings from developed markets.
            Winner of the <strong>ANBIMA Capital Markets Prize</strong>.
        </div>
        <span class="tag">Green Bonds</span>
        <span class="tag">Event Study</span>
        <span class="tag">Sustainable Finance</span>
        <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6692059"
           target="_blank" style="font-size:0.8rem;">Download on SSRN \u2197</a>
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
           target="_blank" style="font-size:0.8rem;">Download on SSRN \u2197</a>
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

with tab3:
    st.markdown('<div class="section-label">Doctoral Students \u2014 Advisor</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="card-body">
            <strong>Lemuel de Lemos Rom\u00e3o</strong> \u00b7 2023<br>
            <span style="color:#6b7280; font-size:0.85rem;">Fundos de investimento no Brasil</span><br><br>
            <strong>Ir\u00e3 In\u00e1cio Ribeiro</strong> \u00b7 2023<br>
            <span style="color:#6b7280; font-size:0.85rem;">Conex\u00f5es pol\u00edticas e precifica\u00e7\u00e3o de ativos: evid\u00eancias no mercado regulado brasileiro</span><br><br>
            <strong>Lu\u00eds Othon Bastos</strong> \u00b7 2022<br>
            <span style="color:#6b7280; font-size:0.85rem;">O Frete de Retorno nas Ferrovias Regionais de Cargas Brasileiras</span><br><br>
            <strong>Robson G\u00f3es de Carvalho</strong> \u00b7 2021<br>
            <span style="color:#6b7280; font-size:0.85rem;">Fam\u00edlia de \u00edndices de renda fixa: uma proposta metodol\u00f3gica</span><br><br>
            <strong>Giovanna Tonetto Segantini</strong> \u00b7 2019<br>
            <span style="color:#6b7280; font-size:0.85rem;">Two essays about mandatory dividend: what does the mandatory dividend have to inform to the market</span><br><br>
            <strong>Melqu\u00edades Pereira de Lima J\u00fanior</strong> \u00b7 2014<br>
            <span style="color:#6b7280; font-size:0.85rem;">Desempenho de analistas sell-side no mercado de a\u00e7\u00f5es brasileiro</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-label">Master\u2019s Students \u2014 Advisor</div>', unsafe_allow_html=True)
    rows = [
        ("Thiago Wanderley Macedo Neves de Almeida", "2022", "Observa\u00e7\u00f5es e infer\u00eancias sobre o custo fixo na estrat\u00e9gia de put protetora"),
        ("Rodrigo Raposo da Fonseca", "2018", "A rela\u00e7\u00e3o entre otimismo e ondas de fus\u00e3o e aquisi\u00e7\u00e3o: evid\u00eancias do mercado brasileiro"),
        ("Heric Nero Lisboa dos Santos", "2018", "Compra de op\u00e7\u00f5es como alternativa para prote\u00e7\u00e3o de carteiras de a\u00e7\u00f5es"),
        ("Aletheia Januaria Zanow de Gouvea", "2016", "Insider Trading no Mercado de Capitais Brasileiro: O Crime Compensa?"),
        ("Patricia Ribeiro Romano", "2015", "Fus\u00f5es e aquisi\u00e7\u00f5es no Brasil: an\u00e1lise dos efeitos em mercado de capitais"),
        ("Tanite de Melo Silva", "2015", "Desempenho de fundos de investimento"),
        ("Ruan Rodrigo Ara\u00fajo da Costa", "2013", "A rela\u00e7\u00e3o entre desempenho e a forma legal das institui\u00e7\u00f5es de microcr\u00e9dito"),
        ("Luiz Carlos Santos J\u00fanior", "2012", "An\u00e1lise experimental do efeito diversifica\u00e7\u00e3o em carteiras de a\u00e7\u00f5es"),
        ("Lana Viviane Linhares da Costa Silva", "2012", "Teoria de carteiras aplicada \u00e0 an\u00e1lise de disposi\u00e7\u00e3o geogr\u00e1fica de usinas e\u00f3licas offshore"),
        ("Jo\u00e3o Paulo Costa de Medeiros", "2012", "Precifica\u00e7\u00e3o da energia e\u00f3lica offshore no Brasil"),
    ]
    body = "<br><br>".join(
        f"<strong>{n}</strong> \u00b7 {y}<br><span style='color:#6b7280; font-size:0.85rem;'>{t}</span>"
        for n, y, t in rows
    )
    st.markdown(f'<div class="card"><div class="card-body">{body}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Vinicio Almeida \u00b7 DEPAD/PPGA \u00b7 UFRN</div>', unsafe_allow_html=True)
