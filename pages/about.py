import streamlit as st

st.markdown('<div class="page-title">Prof. Vinicio Almeida</div>', unsafe_allow_html=True)

# ── Bio header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="font-size:1.02rem; color:#3d4255; line-height:1.7; margin-bottom:1.2rem;">
    Associate Professor of Finance · <strong>UFRN/PPGA</strong><br>
    <span style="font-size:0.88rem; color:#8b8fa8;">
        Graduate School of Business — Federal University at Rio Grande do Norte<br>
        Natal, Rio Grande do Norte, Brazil
    </span>
</div>
""", unsafe_allow_html=True)

# ── Social / links row ────────────────────────────────────────────────────────
st.markdown("""
<div class="social-row">
    <a href="mailto:vinicio.almeida@ufrn.br">vinicio.almeida@ufrn.br</a>
    <a href="https://sigaa.ufrn.br/sigaa/public/docente/portal.jsf?siape=1802347" target="_blank">UFRN</a>
    <a href="https://scholar.google.com.br/citations?user=S-KEieUAAAAJ&hl=pt-BR" target="_blank">Google Scholar</a>
    <a href="http://lattes.cnpq.br/5861723290897089" target="_blank">Lattes CV</a>
    <a href="https://www.linkedin.com/in/vinicioalmeida" target="_blank">LinkedIn</a>
    <a href="https://github.com/vinicioalmeida" target="_blank">GitHub</a>
    <a href="https://www.youtube.com/@almeidavinicio" target="_blank">YouTube</a>
    <a href="https://twitter.com/ovinicioalmeida" target="_blank">Twitter/X</a>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Highlights — 3 cards ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">📘 Book: Guia do Investidor Enganado</div>
        <div class="card-body">
            A guide for retail investors on conflicts of interest, structured products,
            and the traps of the Brazilian financial market.
        </div>
        <a href="https://www.amazon.com.br/Guia-Investidor-Enganado-Conflitos-Interesses-ebook/dp/B0FFHMWSKF"
           target="_blank" style="font-size:0.8rem;">Buy on Amazon ↗</a>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🐍 Course: Derivatives & Python</div>
        <div class="card-body">
            Options pricing, Greeks, Monte Carlo simulation, and volatility analysis
            implemented in Python. Taught in Portuguese.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/course.py", label="Course page ↗")

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">🔬 Research Group in Finance</div>
        <div class="card-body">
            GPFin — DEPAD/UFRN. Research in capital markets, quantitative finance,
            and sustainable finance. Registered at the CNPq Directory of Research Groups.
        </div>
        <a href="http://dgp.cnpq.br/dgp/espelhogrupo/2623"
           target="_blank" style="font-size:0.8rem;">CNPq DGP ↗</a>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Projects & Communities ────────────────────────────────────────────────────
st.markdown('<div class="section-label">Projects & Communities</div>', unsafe_allow_html=True)

pc1, pc2 = st.columns(2)

with pc1:
    st.markdown("""
    <div class="card">
        <div class="card-title">📈 UFRN Investment League</div>
        <div class="card-body">
            Student-run investment league at UFRN, coordinated by Prof. Almeida.
            Weekly sessions covering equity analysis, portfolio management, and
            preparation for national competitions (BTG Challenge, Itaú Quantamental).
        </div>
        <a href="https://www.linkedin.com/company/ufrnliga/" target="_blank"
           style="font-size:0.8rem;">LinkedIn page ↗</a>
    </div>
    """, unsafe_allow_html=True)

with pc2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🤖 @pairsTradingBot</div>
        <div class="card-body">
            Twitter/X bot publishing pairs trading signals based on cointegration
            analysis of Brazilian equities. An open quantitative finance project.
        </div>
        <a href="https://x.com/pairsTradingBot" target="_blank"
           style="font-size:0.8rem;">Follow on X ↗</a>
    </div>
    """, unsafe_allow_html=True)

# ── Areas of interest ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Areas of Interest</div>', unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom:1.2rem;">
    <span class="tag">Derivatives</span>
    <span class="tag">Portfolio Management</span>
    <span class="tag">Quantitative Trading</span>
    <span class="tag">Sell-side Analysts</span>
    <span class="tag">Sustainable Finance</span>
    <span class="tag">Machine Learning in Finance</span>
    <span class="tag">Financial Education</span>
</div>
""", unsafe_allow_html=True)

# ── Education ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Education</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <div class="card-body">
        <strong>D.Sc. in Business Administration (Finance)</strong><br>
        Coppead Graduate School of Business / Federal University at Rio de Janeiro — UFRJ · 2010<br><br>
        <strong>B.S. in Business Administration</strong><br>
        State University at Ceará — UECE · 2005
    </div>
</div>
""", unsafe_allow_html=True)

# ── Employment ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Employment</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <div class="card-body">
        <strong>Federal University at Rio Grande do Norte — UFRN</strong> · 2010–present<br>
        Professor of Finance · DEPAD / PPGA ·
        <a href="https://sigaa.ufrn.br/sigaa/public/docente/portal.jsf?siape=1802347"
           target="_blank" style="font-size:0.88rem;">Official UFRN page ↗</a><br><br>
        <strong>Banco do Brasil S.A.</strong> · 2000–2010<br>
        Portfolio Management · Investment Banking · Financial Distress and Recovery · International Finance
    </div>
</div>
""", unsafe_allow_html=True)


# ── Media & Press ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Media & Press</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <div class="card-title">📰 Investing.com · Opinion</div>
    <div class="card-body">
        Regular contributor to Investing.com Brasil, publishing articles on capital markets,
        derivatives, and financial literacy.
    </div>
    <a href="https://br.investing.com/members/contributors/211720210/opinion"
       target="_blank" style="font-size:0.8rem;">Read articles ↗</a>
</div>
""", unsafe_allow_html=True)

# ── Awards & Certifications ───────────────────────────────────────────────────
st.markdown('<div class="section-label">Awards, Certifications & Grants</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <div class="card-body">
        Certified Portfolio Manager, <em>Comissão de Valores Mobiliários</em> — CVM · 2016<br><br>
        Best Publication, National Association of Capital Markets Analysts and Investment Professionals (APIMEC), with Melquíades Júnior · 2014<br><br>
        Grant on Investments Research, <em>Conselho Nacional de Desenvolvimento Científico e Tecnológico</em> — CNPq · 2012<br><br>
        Best Thesis Project, National Association of Investment Banks (ANBIMA) · 2008
    </div>
</div>
""", unsafe_allow_html=True)

# ── Other scholarly ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Other Scholarly Activities</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <div class="card-body">
        <strong>UCLA Anderson School of Management</strong> · Research Scholar · 2015<br><br>
        <strong>Reviewer</strong> at <em>Academia-Revista Latinoamericana de Administracion</em>,
        <em>Applied Financial Economics</em>, <em>Brazilian Business Review</em>,
        <em>Brazilian Review of Finance</em>, <em>Revista de Administração Mackenzie</em>,
        <em>The Quarterly Review of Economics and Finance</em>, and others.<br><br>
        <strong>Ad-hoc reviewer and presenter</strong> at AIB Annual Meeting, Balas Annual Conference,
        Enanpad Meeting, Encontro Brasileiro de Finanças, Midwest Finance Association Annual Meeting,
        EFMA Annual Meeting.<br><br>
        <strong>Speaker</strong> at seminars on investments at Brazilian Air Force, FGV, UFRJ/Coppead,
        UFRN, and other institutions.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Personal ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Personal</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <div class="card-body">
        Brazilian, born 17 March 1980. Based in Natal, Rio Grande do Norte, Brazil.<br><br>
        Outside academia: rock climbing (active member of AERN — Associação de Escaladores do RN) and guitar playing.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Vinicio Almeida · DEPAD/PPGA · UFRN · Natal, RN · Brazil
</div>
""", unsafe_allow_html=True)
