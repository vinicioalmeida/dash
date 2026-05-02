import streamlit as st

st.markdown('<div class="page-title">Derivatives & Python</div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #0d1b2a 0%, #1a3550 100%);
            border-radius: 7px; padding: 2rem 2.2rem; margin-bottom: 2rem;">
    <div style="font-family: 'Lora', serif; font-size: 1.6rem; font-weight: 600;
                color: #f5f0e8; margin-bottom: 0.5rem;">
        Curso Derivativos &amp; Python
    </div>
    <div style="font-size: 0.95rem; color: #b8c8d8; line-height: 1.7;">
        Online &amp; Ao Vivo &nbsp;·&nbsp; ⏰ 18 horas
    </div>
    <div style="margin-top: 1rem; font-size: 0.82rem; color: #8ca0b8;">
        Taught in Portuguese &nbsp;·&nbsp; 🇧🇷
    </div>
</div>
""", unsafe_allow_html=True)

# ── About ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">About the Course</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <div class="card-body">
        O uso de derivativos financeiros ainda é tema sensível para gestão de carteiras e adoção
        de estratégias de operações em mercado de capitais. Neste curso, discutimos o funcionamento
        desse mercado, seus principais produtos, estratégias de operações e procedimentos de
        precificação.<br><br>
        Haverá exposição dos principais conceitos relacionados à estrutura de funcionamento do
        mercado de derivativos e das maneiras pelas quais esse mercado pode ser utilizado tanto
        para gerenciamento de risco como para implementação de estratégias que visem alavancar
        retornos em apostas direcionais. Serão tratados os fatores que afetam os preços de opções
        e futuros e os métodos de avaliação pelo modelo binomial e pelo modelo de
        Black &amp; Scholes.<br><br>
        Todo o curso é realizado com demonstrações de exemplos e implementações em Python.
    </div>
</div>
""", unsafe_allow_html=True)

# ── What you get ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">O que você terá</div>', unsafe_allow_html=True)
st.markdown("""
<div class="card">
    <div class="card-body">
        ✔️ Aulas expositivas ao vivo com o professor e práticas durante todo o curso.<br>
        ✔️ Todos os arquivos utilizados durante o curso, incluindo os códigos em Python.<br>
        ✔️ Apoio em projeto individual de implementação de estratégia em Python.<br><br>
        As sessões são conduzidas ao vivo via <strong>Google Meet</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Syllabus ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Conteúdo — 9 sessões</div>', unsafe_allow_html=True)

sessions = [
    ("Dia 01", "Introdução ao mercado de derivativos e seus principais produtos"),
    ("Dia 02", "Funcionamento do mercado de futuros e contratos a termo"),
    ("Dia 03", "Estratégias de hedge e arbitragem usando futuros"),
    ("Dia 04", "Funcionamento e fatores que afetam os preços das opções"),
    ("Dia 05", "Operações estruturadas com opções"),
    ("Dia 06", "Árvores binomiais e Modelo de Black-Scholes-Merton"),
    ("Dia 07", "Letras gregas"),
    ("Dia 08", "Smile de volatilidade"),
    ("Dia 09", "Opções exóticas"),
]

col1, col2 = st.columns(2)
for i, (dia, topico) in enumerate(sessions):
    col = col1 if i % 2 == 0 else col2
    with col:
        st.markdown(f"""
        <div class="card" style="padding: 0.75rem 1rem;">
            <span style="font-size:0.72rem; color:#b8973a; font-weight:600;
                         letter-spacing:0.05em; text-transform:uppercase;">{dia}</span>
            <div class="card-body" style="margin-top:0.2rem;">{topico}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Tags ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top: 0.5rem; margin-bottom: 2rem;">
    <span class="tag">Derivativos</span>
    <span class="tag">Opções</span>
    <span class="tag">Futuros</span>
    <span class="tag">Black-Scholes</span>
    <span class="tag">Gregas</span>
    <span class="tag">Volatilidade</span>
    <span class="tag">Python</span>
    <span class="tag">Hedge</span>
    <span class="tag">Arbitragem</span>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="footer">Vinicio Almeida · DEPAD/PPGA · UFRN</div>', unsafe_allow_html=True)
