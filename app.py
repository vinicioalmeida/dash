import streamlit as st

# ── Page registry ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vinicio Almeida",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

about      = st.Page("pages/about.py",      title="About",            icon="👤", default=True)
research   = st.Page("pages/research.py",   title="Research",         icon="📄")
teaching   = st.Page("pages/teaching.py",   title="Teaching",         icon="🎓")
course     = st.Page("pages/course.py",     title="Derivatives & Python", icon="🐍")
tools      = st.Page("pages/tools.py",      title="Tools",            icon="🛠️")

pg = st.navigation(
    {
        "": [about],
        "Academic": [research, teaching],
        "Resources": [course, tools],
    },
    expanded=True,
)

# ── Shared CSS injected on every page ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1a1a2e;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0d1b2a !important;
}
section[data-testid="stSidebar"] * {
    color: #cdd8e3 !important;
}
section[data-testid="stSidebar"] .st-emotion-cache-1rtdyuf,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    font-size: 0.88rem;
}

/* Main content */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 860px;
}

/* Page title */
.page-title {
    font-family: 'Lora', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #0d1b2a;
    border-bottom: 2px solid #b8973a;
    padding-bottom: 0.35rem;
    margin-bottom: 1.8rem;
    letter-spacing: -0.01em;
}

/* Section label */
.section-label {
    font-family: 'Lora', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #0d1b2a;
    margin-top: 2rem;
    margin-bottom: 0.7rem;
    border-left: 3px solid #b8973a;
    padding-left: 0.55rem;
}

/* Card */
.card {
    background: #ffffff;
    border: 1px solid #e5e0d8;
    border-radius: 5px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.85rem;
}
.card-title {
    font-family: 'Lora', serif;
    font-size: 0.98rem;
    font-weight: 600;
    color: #0d1b2a;
    margin-bottom: 0.2rem;
}
.card-meta {
    font-size: 0.79rem;
    color: #8b8fa8;
    margin-bottom: 0.35rem;
    letter-spacing: 0.01em;
}
.card-body {
    font-size: 0.88rem;
    color: #3d4255;
    line-height: 1.6;
}

/* Tag */
.tag {
    display: inline-block;
    background: #0d1b2a;
    color: #cdd8e3 !important;
    font-size: 0.7rem;
    border-radius: 3px;
    padding: 2px 7px;
    margin-right: 4px;
    margin-top: 6px;
    letter-spacing: 0.02em;
}

/* Social link row */
.social-row a {
    display: inline-block;
    margin-right: 1rem;
    font-size: 0.83rem;
    color: #1a3550;
    text-decoration: none;
    border-bottom: 1px solid #b8973a;
    padding-bottom: 1px;
}
.social-row a:hover { color: #b8973a; }

/* Footer */
.footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #e5e0d8;
    font-size: 0.77rem;
    color: #adb5bd;
    text-align: center;
}

a { color: #1a3550; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

pg.run()
