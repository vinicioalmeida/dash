import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# ── Language notice ──────────────────────────────────────────────────────────
st.markdown('<div class="page-title">Quantitative Tools</div>', unsafe_allow_html=True)

st.info("🇧🇷 **These tools are in Portuguese.**")

st.markdown("---")

# ── Tool selector ─────────────────────────────────────────────────────────────
TOOLS = [
    "Calculadora de VPL",
    "Estrutura de Capital",
    "Dividendos vs JCP",
    "Títulos de Renda Fixa",
    "Economia do 'e'",
    "Análise de Risco e Retorno",
    "Black-Scholes-Merton",
    "Gregas de Opções",
    "Payoff de Opções",
    "Simulador de Monte Carlo",
    "Hedge Cambial",
]

selected_tool = st.selectbox("Escolha a Ferramenta:", TOOLS)
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════════════════

def black_scholes(S, K, T, r, sigma, tipo="call"):
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if tipo.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def monte_carlo_option_pricing(S, K, T, r, sigma, n_sims, n_steps, tipo="call"):
    dt = T / n_steps
    np.random.seed(42)
    Z = np.random.standard_normal((n_sims, n_steps))
    S_paths = np.zeros((n_sims, n_steps + 1))
    S_paths[:, 0] = S
    for t in range(1, n_steps + 1):
        S_paths[:, t] = S_paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )
    S_final = S_paths[:, -1]
    if tipo.lower() == "call":
        payoffs = np.maximum(S_final - K, 0)
    else:
        payoffs = np.maximum(K - S_final, 0)
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price, S_paths, S_final, payoffs


def calcular_vpl(investimento_inicial, fluxos_caixa, taxa_desconto):
    vp_fluxos = []
    vp_total = 0
    for t, fluxo in enumerate(fluxos_caixa, 1):
        vp_fluxo = fluxo / ((1 + taxa_desconto) ** t)
        vp_fluxos.append(vp_fluxo)
        vp_total += vp_fluxo
    vpl = investimento_inicial + vp_total
    return vpl, vp_total, vp_fluxos


def taxa_interna_retorno(investimento_inicial, fluxos_caixa, tentativas=1000):
    def vpl_para_taxa(taxa):
        return calcular_vpl(investimento_inicial, fluxos_caixa, taxa)[0]
    taxa_min, taxa_max = -0.99, 10.0
    for _ in range(tentativas):
        taxa_media = (taxa_min + taxa_max) / 2
        vpl_medio = vpl_para_taxa(taxa_media)
        if abs(vpl_medio) < 0.01:
            return taxa_media
        elif vpl_medio > 0:
            taxa_min = taxa_media
        else:
            taxa_max = taxa_media
    return taxa_media


def calcular_preco_titulo(valor_face, cupom, ytm, maturidade, freq_cupom=1):
    if freq_cupom == 1:
        periodos = int(maturidade)
        taxa_periodo = ytm
        cupom_periodo = cupom
    else:
        periodos = int(maturidade * freq_cupom)
        taxa_periodo = ytm / freq_cupom
        cupom_periodo = cupom / freq_cupom
    if periodos == 0:
        return valor_face
    vp_cupons = sum(cupom_periodo / ((1 + taxa_periodo) ** t) for t in range(1, periodos + 1))
    vp_principal = valor_face / ((1 + taxa_periodo) ** periodos)
    return vp_cupons + vp_principal


def calcular_ytm_titulo(preco, valor_face, cupom, maturidade, freq_cupom=1, tentativas=1000):
    ytm_min, ytm_max = 0.001, 1.0
    for _ in range(tentativas):
        ytm_medio = (ytm_min + ytm_max) / 2
        preco_calculado = calcular_preco_titulo(valor_face, cupom, ytm_medio, maturidade, freq_cupom)
        if abs(preco_calculado - preco) < 0.01:
            return ytm_medio
        elif preco_calculado > preco:
            ytm_min = ytm_medio
        else:
            ytm_max = ytm_medio
    return ytm_medio


def calcular_duration_macaulay(valor_face, cupom, ytm, maturidade, freq_cupom=1):
    if freq_cupom == 1:
        periodos = int(maturidade)
        taxa_periodo = ytm
        cupom_periodo = cupom
    else:
        periodos = int(maturidade * freq_cupom)
        taxa_periodo = ytm / freq_cupom
        cupom_periodo = cupom / freq_cupom
    preco_titulo = calcular_preco_titulo(valor_face, cupom, ytm, maturidade, freq_cupom)
    duration_numerador = sum(
        t * (cupom_periodo / ((1 + taxa_periodo) ** t)) for t in range(1, periodos + 1)
    )
    vp_principal = valor_face / ((1 + taxa_periodo) ** periodos)
    duration_numerador += periodos * vp_principal
    duration = duration_numerador / preco_titulo
    if freq_cupom > 1:
        duration = duration / freq_cupom
    return duration


def calcular_duration_modificada(duration_macaulay, ytm, freq_cupom=1):
    return duration_macaulay / (1 + ytm / freq_cupom)


def calcular_convexidade(valor_face, cupom, ytm, maturidade, freq_cupom=1):
    if freq_cupom == 1:
        periodos = int(maturidade)
        taxa_periodo = ytm
        cupom_periodo = cupom
    else:
        periodos = int(maturidade * freq_cupom)
        taxa_periodo = ytm / freq_cupom
        cupom_periodo = cupom / freq_cupom
    preco_titulo = calcular_preco_titulo(valor_face, cupom, ytm, maturidade, freq_cupom)
    conv_num = sum(
        t * (t + 1) * (cupom_periodo / ((1 + taxa_periodo) ** t)) for t in range(1, periodos + 1)
    )
    vp_principal = valor_face / ((1 + taxa_periodo) ** periodos)
    conv_num += periodos * (periodos + 1) * vp_principal
    convexidade = conv_num / (preco_titulo * ((1 + taxa_periodo) ** 2))
    if freq_cupom > 1:
        convexidade = convexidade / (freq_cupom**2)
    return convexidade


def calcular_current_yield(cupom_anual, preco_mercado):
    return cupom_anual / preco_mercado


def estimar_variacao_preco(duration_modificada, convexidade, variacao_ytm):
    return -duration_modificada * variacao_ytm + 0.5 * convexidade * (variacao_ytm**2)


def calcular_capm(rf, beta, rm):
    return rf + beta * (rm - rf)


def calcular_wacc(E, D, V, Re, Rd, tax_rate):
    return (E / V) * Re + (D / V) * Rd * (1 - tax_rate)


def analisar_estrutura_otima(valores_divida, patrimonio_liquido, custo_equity, custo_divida, tax_rate, custo_falencia_rate=0.02):
    resultados = []
    for D in valores_divida:
        V = D + patrimonio_liquido
        debt_ratio = D / V
        tax_shield = tax_rate * D
        custo_falencia = custo_falencia_rate * debt_ratio**2 * V
        wacc = calcular_wacc(patrimonio_liquido, D, V, custo_equity, custo_divida, tax_rate)
        valor_empresa = V + tax_shield - custo_falencia
        resultados.append({
            "Divida": D, "Patrimonio_Liquido": patrimonio_liquido, "Valor_Total": V,
            "Debt_Ratio": debt_ratio, "Equity_Ratio": patrimonio_liquido / V,
            "Tax_Shield": tax_shield, "Custo_Falencia": custo_falencia,
            "WACC": wacc, "Valor_Empresa": valor_empresa,
        })
    return pd.DataFrame(resultados)


def calcular_limite_jcp(patrimonio_liquido, tjlp_anual):
    return patrimonio_liquido * tjlp_anual


def calcular_tax_shield_jcp(valor_jcp, aliquota_ir=0.25, aliquota_csll=0.09):
    return valor_jcp * (aliquota_ir + aliquota_csll)


def calcular_recebimento_liquido_jcp(valor_jcp, ir_retido=0.15):
    return valor_jcp * (1 - ir_retido)


def analisar_dividendos_vs_jcp(valor_distribuicao, patrimonio_liquido, tjlp_anual, lucro_liquido,
                                aliquota_ir=0.25, aliquota_csll=0.09, ir_retido_jcp=0.15):
    limite_jcp = calcular_limite_jcp(patrimonio_liquido, tjlp_anual)
    dividendos = valor_distribuicao
    liquido_acionista_div = dividendos
    custo_empresa_div = dividendos
    jcp_possivel = min(valor_distribuicao, limite_jcp)
    tax_shield_jcp = calcular_tax_shield_jcp(jcp_possivel, aliquota_ir, aliquota_csll)
    liquido_acionista_jcp = calcular_recebimento_liquido_jcp(jcp_possivel, ir_retido_jcp)
    custo_empresa_jcp = jcp_possivel - tax_shield_jcp
    dividendo_complementar = valor_distribuicao - jcp_possivel
    if dividendo_complementar > 0:
        liquido_acionista_misto = liquido_acionista_jcp + dividendo_complementar
        custo_empresa_misto = custo_empresa_jcp + dividendo_complementar
        tax_shield_misto = tax_shield_jcp
    else:
        liquido_acionista_misto = liquido_acionista_jcp
        custo_empresa_misto = custo_empresa_jcp
        tax_shield_misto = tax_shield_jcp
    return {
        "valor_distribuicao": valor_distribuicao, "limite_jcp": limite_jcp,
        "jcp_possivel": jcp_possivel, "dividendo_complementar": dividendo_complementar,
        "cenario_dividendos": {"valor": dividendos, "tax_shield": 0,
                               "liquido_acionista": liquido_acionista_div, "custo_empresa": custo_empresa_div},
        "cenario_jcp": {"valor": jcp_possivel, "tax_shield": tax_shield_jcp,
                        "liquido_acionista": liquido_acionista_jcp, "custo_empresa": custo_empresa_jcp},
        "cenario_misto": {"jcp": jcp_possivel, "dividendos": dividendo_complementar,
                          "tax_shield": tax_shield_misto, "liquido_acionista": liquido_acionista_misto,
                          "custo_empresa": custo_empresa_misto},
    }


def calcular_e_aproximacao(m):
    return (1 + 1 / m) ** m


def valor_futuro_discreto(principal, taxa, tempo):
    return principal * (1 + taxa) ** tempo


def valor_futuro_continuo(principal, taxa, tempo):
    return principal * np.exp(taxa * tempo)


def valor_futuro_composto(principal, taxa_nominal, freq_capitalizacao, tempo):
    return principal * (1 + taxa_nominal / freq_capitalizacao) ** (freq_capitalizacao * tempo)


def taxa_discreta_para_continua(taxa_discreta):
    return np.log(1 + taxa_discreta)


def taxa_continua_para_discreta(taxa_continua):
    return np.exp(taxa_continua) - 1


def valor_presente_discreto(valor_futuro, taxa, tempo):
    return valor_futuro / (1 + taxa) ** tempo


def valor_presente_continuo(valor_futuro, taxa, tempo):
    return valor_futuro * np.exp(-taxa * tempo)


def crescimento_populacional(populacao_inicial, taxa_crescimento, tempo):
    return populacao_inicial * np.exp(taxa_crescimento * tempo)


def gerar_dados_exemplo(dias=500):
    np.random.seed(42)
    ativos = {
        "IBOV": {"ret_medio": 0.0008, "vol": 0.020},
        "PETR4": {"ret_medio": 0.0010, "vol": 0.035},
        "VALE3": {"ret_medio": 0.0012, "vol": 0.030},
        "ITUB4": {"ret_medio": 0.0006, "vol": 0.025},
    }
    correlacoes = np.array([
        [1.00, 0.75, 0.65, 0.60],
        [0.75, 1.00, 0.45, 0.50],
        [0.65, 0.45, 1.00, 0.40],
        [0.60, 0.50, 0.40, 1.00],
    ])
    retornos_independentes = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=correlacoes, size=dias)
    dados = {}
    precos = {}
    for i, (nome, params) in enumerate(ativos.items()):
        retornos = params["ret_medio"] + params["vol"] * retornos_independentes[:, i]
        dados[f"{nome}_ret"] = retornos
        preco_inicial = 100 if nome == "IBOV" else np.random.uniform(10, 50)
        precos_ativo = [preco_inicial]
        for ret in retornos[1:]:
            precos_ativo.append(precos_ativo[-1] * (1 + ret))
        precos[nome] = precos_ativo
        dados[nome] = precos_ativo
    dates = pd.date_range(start="2022-01-01", periods=dias, freq="D")
    df_precos = pd.DataFrame(precos, index=dates)
    df_retornos = pd.DataFrame(
        {k: v for k, v in dados.items() if "_ret" in k}, index=dates
    )
    df_retornos.columns = [col.replace("_ret", "") for col in df_retornos.columns]
    return df_precos, df_retornos


def calcular_estatisticas_risco_retorno(retornos):
    stats = {}
    for col in retornos.columns:
        serie = retornos[col]
        stats[col] = {
            "Retorno Médio Diário (%)": serie.mean() * 100,
            "Retorno Anualizado (%)": serie.mean() * 252 * 100,
            "Volatilidade Diária (%)": serie.std() * 100,
            "Volatilidade Anualizada (%)": serie.std() * np.sqrt(252) * 100,
            "Sharpe Ratio": (serie.mean() * 252) / (serie.std() * np.sqrt(252)) if serie.std() > 0 else 0,
            "Máximo (%)": serie.max() * 100,
            "Mínimo (%)": serie.min() * 100,
            "Assimetria": serie.skew(),
            "Curtose": serie.kurtosis(),
        }
    return stats


def calcular_carteira_otima(retornos_A, retornos_B, rf=0.0):
    ret_A = retornos_A.mean() * 252
    ret_B = retornos_B.mean() * 252
    vol_A = retornos_A.std() * np.sqrt(252)
    vol_B = retornos_B.std() * np.sqrt(252)
    corr = retornos_A.corr(retornos_B)
    numerador = (ret_A - rf) * vol_B**2 - (ret_B - rf) * vol_A * vol_B * corr
    denominador = (ret_A - rf) * vol_B**2 + (ret_B - rf) * vol_A**2 - (ret_A - rf + ret_B - rf) * vol_A * vol_B * corr
    w_A = 0.5 if abs(denominador) < 1e-10 else numerador / denominador
    w_A = max(0, min(1, w_A))
    return w_A, 1 - w_A


def fronteira_eficiente_dois_ativos(retornos_A, retornos_B, pontos=100):
    pesos = np.linspace(0, 1, pontos)
    ret_A = retornos_A.mean() * 252
    ret_B = retornos_B.mean() * 252
    vol_A = retornos_A.std() * np.sqrt(252)
    vol_B = retornos_B.std() * np.sqrt(252)
    corr = retornos_A.corr(retornos_B)
    retornos_carteira, riscos_carteira = [], []
    for w_A in pesos:
        w_B = 1 - w_A
        ret_cart = w_A * ret_A + w_B * ret_B
        var_cart = w_A**2 * vol_A**2 + w_B**2 * vol_B**2 + 2 * w_A * w_B * vol_A * vol_B * corr
        retornos_carteira.append(ret_cart)
        riscos_carteira.append(np.sqrt(var_cart))
    return pesos, np.array(retornos_carteira), np.array(riscos_carteira)


# ════════════════════════════════════════════════════════════════════════════
# Tool rendering — original logic preserved exactly
# ════════════════════════════════════════════════════════════════════════════

if selected_tool == "Calculadora de VPL":
    st.subheader("Calculadora de Valor Presente Líquido (VPL)")
    st.write("""
    O **Valor Presente Líquido (VPL)** é uma ferramenta importante para avaliação de projetos de investimento.

    **Equação:** VPL = C₀ + Σ[Cₜ/(1+r)ᵗ]

    Onde:
    - C₀ = Investimento inicial (fluxo negativo)
    - Cₜ = Fluxo de caixa no período t
    - r = Taxa de desconto (custo de capital)
    - t = Período
    """)

    tab1, tab2, tab3 = st.tabs(["Cálculo Básico", "Análise de Sensibilidade", "Múltiplos Projetos"])

    with tab1:
        st.subheader("Cálculo do VPL")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dados do Projeto:**")
            investimento_inicial = st.number_input("Investimento Inicial (R$ mil):", value=100.0, step=10.0)
            taxa_desconto = st.number_input("Taxa de Desconto (% a.a.):", min_value=0.0, max_value=50.0, value=10.0, step=0.5) / 100
            num_periodos = st.slider("Número de Períodos:", min_value=1, max_value=10, value=5)
        with col2:
            st.write("**Fluxos de Caixa Futuros (R$ mil):**")
            fluxos_caixa = [st.number_input(f"Ano {i+1}:", value=30.0, step=5.0, key=f"fluxo_{i}") for i in range(num_periodos)]

        if st.button("Calcular VPL", key="calc_vpl_basic"):
            investimento_negativo = -abs(investimento_inicial)
            vpl, vp_total, vp_fluxos = calcular_vpl(investimento_negativo, fluxos_caixa, taxa_desconto)
            try:
                tir = taxa_interna_retorno(investimento_negativo, fluxos_caixa)
                tir_percent = tir * 100
            except:
                tir_percent = "N/A"

            st.subheader("Resultados da Análise")
            col1, col2, col3 = st.columns(3)
            with col1:
                cor_vpl = "green" if vpl > 0 else "red"
                st.markdown(f'<div style="text-align:center;padding:20px;border:2px solid {cor_vpl};border-radius:10px;"><h3 style="color:{cor_vpl};">VPL</h3><h2 style="color:{cor_vpl};">R$ {vpl:.2f} mil</h2></div>', unsafe_allow_html=True)
            with col2:
                st.metric("VP dos Fluxos", f"R$ {vp_total:.2f} mil")
                st.metric("Investimento Inicial", f"R$ {investimento_inicial:.2f} mil")
            with col3:
                if isinstance(tir_percent, (int, float)):
                    cor_tir = "green" if tir_percent > taxa_desconto * 100 else "red"
                    st.markdown(f'<div style="text-align:center;padding:10px;"><h4>TIR</h4><h3 style="color:{cor_tir};">{tir_percent:.2f}%</h3><small>Taxa de Desconto: {taxa_desconto*100:.1f}%</small></div>', unsafe_allow_html=True)

            if vpl > 0:
                st.success(f"PROJETO VIÁVEL — VPL positivo de R$ {vpl:.2f} mil. Retorno superior ao custo de capital ({taxa_desconto*100:.1f}%). Decisão: ACEITAR.")
            elif vpl < 0:
                st.error(f"PROJETO NÃO VIÁVEL — VPL negativo de R$ {vpl:.2f} mil. Retorno inferior ao custo de capital. Decisão: REJEITAR.")
            else:
                st.warning("PROJETO NEUTRO — VPL = 0.")

            st.subheader("Detalhamento dos Cálculos")
            dados_tabela = [{"Período": 0, "Fluxo de Caixa": f"R$ {investimento_negativo:.2f}", "Fator de Desconto": "1,0000", "Valor Presente": f"R$ {investimento_negativo:.2f}"}]
            for i, (fluxo, vp) in enumerate(zip(fluxos_caixa, vp_fluxos), 1):
                fator = 1 / ((1 + taxa_desconto) ** i)
                dados_tabela.append({"Período": i, "Fluxo de Caixa": f"R$ {fluxo:.2f}", "Fator de Desconto": f"{fator:.4f}", "Valor Presente": f"R$ {vp:.2f}"})
            st.dataframe(pd.DataFrame(dados_tabela), use_container_width=True)

            periodos = list(range(num_periodos + 1))
            fluxos_totais = [investimento_negativo] + fluxos_caixa
            vp_totais = [investimento_negativo] + vp_fluxos
            fig = go.Figure()
            fig.add_trace(go.Bar(x=periodos, y=fluxos_totais, name="Fluxos Nominais", marker_color=["red" if x < 0 else "lightblue" for x in fluxos_totais], opacity=0.7))
            fig.add_trace(go.Bar(x=periodos, y=vp_totais, name="Valores Presentes", marker_color=["darkred" if x < 0 else "darkblue" for x in vp_totais], opacity=0.9))
            fig.update_layout(title="Fluxos de Caixa: Nominais vs Valores Presentes", xaxis_title="Período", yaxis_title="Valor (R$ mil)", template="plotly_dark", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Análise de Sensibilidade")
        taxa_min_s = st.number_input("Taxa Mínima (%)", value=5.0, step=1.0) / 100
        taxa_max_s = st.number_input("Taxa Máxima (%)", value=20.0, step=1.0) / 100
        if st.button("Gerar Análise de Sensibilidade"):
            taxas = np.linspace(taxa_min_s, taxa_max_s, 50)
            vpls = [calcular_vpl(-investimento_inicial, fluxos_caixa, t)[0] for t in taxas]
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(x=taxas * 100, y=vpls, mode="lines", name="VPL", line=dict(width=3, color="blue")))
            fig_sens.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="VPL = 0")
            vpl_atual, _, _ = calcular_vpl(-investimento_inicial, fluxos_caixa, taxa_desconto)
            fig_sens.add_trace(go.Scatter(x=[taxa_desconto * 100], y=[vpl_atual], mode="markers", name=f"Taxa Atual ({taxa_desconto*100:.1f}%)", marker=dict(size=10, color="red")))
            fig_sens.update_layout(title="Análise de Sensibilidade: VPL vs Taxa de Desconto", xaxis_title="Taxa de Desconto (%)", yaxis_title="VPL (R$ mil)", template="plotly_dark")
            st.plotly_chart(fig_sens, use_container_width=True)

    with tab3:
        st.subheader("Comparação de Múltiplos Projetos")
        num_projetos = st.slider("Número de Projetos a Comparar:", 2, 5, 3)
        taxa_comparacao = st.number_input("Taxa de Desconto para Comparação (%):", value=10.0) / 100
        projetos_dados = []
        for i in range(num_projetos):
            st.write(f"**Projeto {chr(65+i)}:**")
            c1, c2 = st.columns(2)
            with c1:
                inv = st.number_input("Investimento Inicial (R$ mil):", value=100.0, key=f"inv_{i}")
                per = st.slider("Períodos:", 1, 8, 5, key=f"per_{i}")
            with c2:
                flx = [st.number_input(f"Ano {j+1}:", value=30.0, key=f"fluxo_{i}_{j}") for j in range(per)]
            projetos_dados.append({"nome": f"Projeto {chr(65+i)}", "investimento": inv, "fluxos": flx})

        if st.button("Comparar Projetos"):
            res = []
            for p in projetos_dados:
                vpl, vp_t, _ = calcular_vpl(-p["investimento"], p["fluxos"], taxa_comparacao)
                try:
                    tir = taxa_interna_retorno(-p["investimento"], p["fluxos"]) * 100
                except:
                    tir = "N/A"
                il = vp_t / p["investimento"] if p["investimento"] > 0 else 0
                res.append({"Projeto": p["nome"], "Investimento (R$ mil)": p["investimento"], "VPL (R$ mil)": vpl, "TIR (%)": f"{tir:.2f}" if isinstance(tir, (int, float)) else tir, "IL": f"{il:.2f}", "Decisão": "ACEITAR" if vpl > 0 else "REJEITAR"})
            st.dataframe(pd.DataFrame(res), use_container_width=True)
            vpls_comp = [r["VPL (R$ mil)"] for r in res]
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(x=[r["Projeto"] for r in res], y=vpls_comp, marker_color=["green" if v > 0 else "red" for v in vpls_comp], text=[f"R$ {v:.1f}" for v in vpls_comp], textposition="auto"))
            fig_comp.add_hline(y=0, line_dash="dash", line_color="white")
            fig_comp.update_layout(title="Comparação de VPL entre Projetos", xaxis_title="Projetos", yaxis_title="VPL (R$ mil)", template="plotly_dark")
            st.plotly_chart(fig_comp, use_container_width=True)
            melhor = max(res, key=lambda x: x["VPL (R$ mil)"])
            if melhor["VPL (R$ mil)"] > 0:
                st.success(f"Recomendação: {melhor['Projeto']} — maior VPL: R$ {melhor['VPL (R$ mil)']:.2f} mil")
            else:
                st.warning("Nenhum projeto apresenta VPL positivo.")

elif selected_tool == "Black-Scholes-Merton":
    st.subheader("Calculadora Black-Scholes-Merton")
    S = st.number_input("Preço do ativo subjacente (S):", min_value=0.0, step=0.01)
    K = st.number_input("Preço de exercício (K):", min_value=0.0, step=0.01)
    T = st.number_input("Tempo até o vencimento (T) em anos:", min_value=0.0, step=0.01)
    r = st.number_input("Taxa livre de risco (r) em %:", min_value=0.0, step=0.01) / 100
    sigma = st.number_input("Volatilidade (σ) em %:", min_value=0.0, step=0.01) / 100
    tipo_opcao = st.radio("Tipo de opção:", ["Call", "Put"])
    if st.button("Calcular"):
        valor_opcao = black_scholes(S, K, T, r, sigma, tipo_opcao)
        st.write(f"**Preço da opção {tipo_opcao}:** R$ {valor_opcao:.2f}")

elif selected_tool == "Gregas de Opções":
    st.subheader("Calculadora de Gregas de Opções")
    S = st.number_input("Preço do ativo subjacente (S):", min_value=0.0, step=0.01, key="S_g")
    K = st.number_input("Preço de exercício (K):", min_value=0.0, step=0.01, key="K_g")
    T = st.number_input("Tempo até o vencimento (T) em anos:", min_value=0.0, step=0.01, key="T_g")
    r = st.number_input("Taxa livre de risco (r) em %:", min_value=0.0, step=0.01, key="r_g") / 100
    sigma = st.number_input("Volatilidade (σ) em %:", min_value=0.0, step=0.01, key="sigma_g") / 100
    if st.button("Calcular Gregas"):
        d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        st.write(f"**Delta:** {norm.cdf(d1):.4f}")
        st.write(f"**Gamma:** {norm.pdf(d1) / (S * sigma * np.sqrt(T)):.4f}")
        st.write(f"**Theta:** {(-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2):.4f}")
        st.write(f"**Vega:** {S * norm.pdf(d1) * np.sqrt(T):.4f}")
        st.write(f"**Rho:** {K * T * np.exp(-r * T) * norm.cdf(d2):.4f}")

elif selected_tool == "Payoff de Opções":
    st.subheader("Payoff de Opções")
    tipo_opcao = st.selectbox("Escolha o tipo de opção:", ["Compra de Call", "Venda de Call", "Compra de Put", "Venda de Put"])
    strike = st.number_input("Preço de Exercício (Strike):", min_value=0.0, step=0.01)
    premio = st.number_input("Prêmio da Opção:", min_value=0.0, step=0.01)
    preco_ativo = st.slider("Intervalo de preços do ativo subjacente:", min_value=0.0, max_value=200.0, value=(0.0, 100.0), step=1.0)
    precos = np.linspace(preco_ativo[0], preco_ativo[1], 500)
    if tipo_opcao == "Compra de Call":
        payoff = np.maximum(precos - strike, 0) - premio
    elif tipo_opcao == "Venda de Call":
        payoff = premio - np.maximum(precos - strike, 0)
    elif tipo_opcao == "Compra de Put":
        payoff = np.maximum(strike - precos, 0) - premio
    else:
        payoff = premio - np.maximum(strike - precos, 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=precos, y=payoff, mode="lines", name="Payoff"))
    fig.update_layout(title="Gráfico de Payoff da Opção", xaxis_title="Preço do Ativo Subjacente", yaxis_title="Payoff", yaxis=dict(range=[payoff.min() * 2, payoff.max() * 2]), template="plotly_dark")
    st.plotly_chart(fig)

elif selected_tool == "Simulador de Monte Carlo":
    st.subheader("Simulador de Monte Carlo para Opções")
    st.write("**Monte Carlo** é um método numérico que simula milhares de possíveis trajetórias do preço do ativo para calcular o valor da opção.")
    col1, col2 = st.columns(2)
    with col1:
        S_mc = st.number_input("Preço do ativo subjacente (S):", min_value=0.01, value=100.0, step=0.01, key="S_mc")
        K_mc = st.number_input("Preço de exercício (K):", min_value=0.01, value=100.0, step=0.01, key="K_mc")
        T_mc = st.number_input("Tempo até vencimento (T) em anos:", min_value=0.01, value=0.25, step=0.01, key="T_mc")
        r_mc = st.number_input("Taxa livre de risco (r) em %:", min_value=0.0, value=5.0, step=0.01, key="r_mc") / 100
        sigma_mc = st.number_input("Volatilidade (σ) em %:", min_value=0.01, value=20.0, step=0.01, key="sigma_mc") / 100
        tipo_mc = st.radio("Tipo de opção:", ["Call", "Put"], key="tipo_mc")
    with col2:
        n_sims = st.slider("Número de simulações:", min_value=1000, max_value=50000, value=10000, step=1000)
        n_steps = st.slider("Número de passos temporais:", min_value=50, max_value=500, value=100, step=50)
        mostrar_trajetorias = st.checkbox("Mostrar algumas trajetórias", value=True)
        n_trajetorias_plot = st.slider("Trajetórias a mostrar:", min_value=5, max_value=100, value=20, step=5)

    if st.button("Executar Simulação Monte Carlo", key="run_mc"):
        with st.spinner("Executando simulação..."):
            mc_price, paths, final_prices, payoffs = monte_carlo_option_pricing(S_mc, K_mc, T_mc, r_mc, sigma_mc, n_sims, n_steps, tipo_mc)
            bs_price = black_scholes(S_mc, K_mc, T_mc, r_mc, sigma_mc, tipo_mc)
        col1, col2, col3 = st.columns(3)
        col1.metric("Preço Monte Carlo", f"R$ {mc_price:.4f}")
        col2.metric("Preço Black-Scholes", f"R$ {bs_price:.4f}")
        col3.metric("Diferença (%)", f"{((mc_price - bs_price) / bs_price) * 100:.2f}%")
        if mostrar_trajetorias:
            time_steps = np.linspace(0, T_mc, n_steps + 1)
            fig_paths = go.Figure()
            for i in range(min(n_trajetorias_plot, n_sims)):
                fig_paths.add_trace(go.Scatter(x=time_steps, y=paths[i], mode="lines", showlegend=False, line=dict(width=1), opacity=0.6))
            fig_paths.add_hline(y=K_mc, line_dash="dash", line_color="red", annotation_text=f"Strike: R$ {K_mc}")
            fig_paths.update_layout(title="Trajetórias de Preço (GBM)", xaxis_title="Tempo (anos)", yaxis_title="Preço do Ativo", template="plotly_dark", height=500)
            st.plotly_chart(fig_paths, use_container_width=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=final_prices, nbinsx=50, opacity=0.7))
        fig_hist.add_vline(x=K_mc, line_dash="dash", line_color="red", annotation_text=f"Strike: R$ {K_mc}")
        fig_hist.update_layout(title="Distribuição dos Preços Finais", xaxis_title="Preço Final", yaxis_title="Frequência", template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

elif selected_tool == "Análise de Risco e Retorno":
    st.subheader("Teoria de Carteiras: Análise de Risco e Retorno")
    df_precos, df_retornos = gerar_dados_exemplo(dias=500)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dados e Visualização", "Estatísticas", "Correlações", "Fronteira Eficiente", "CAPM"])

    with tab1:
        ativo_g = st.selectbox("Selecione o ativo:", df_precos.columns, key="ativo_preco")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=df_precos.index, y=df_precos[ativo_g], mode="lines", name=ativo_g))
        fig_p.update_layout(title=f"Evolução do Preço — {ativo_g}", xaxis_title="Data", yaxis_title="Preço", template="plotly_dark")
        st.plotly_chart(fig_p, use_container_width=True)

    with tab2:
        stats = calcular_estatisticas_risco_retorno(df_retornos)
        st.dataframe(pd.DataFrame(stats).T.round(4), use_container_width=True)
        fig_sc = go.Figure()
        for ativo in df_retornos.columns:
            ret_a = df_retornos[ativo].mean() * 252 * 100
            vol_a = df_retornos[ativo].std() * np.sqrt(252) * 100
            fig_sc.add_trace(go.Scatter(x=[vol_a], y=[ret_a], mode="markers+text", text=[ativo], textposition="top center", marker=dict(size=12), showlegend=False))
        fig_sc.update_layout(title="Risco × Retorno", xaxis_title="Volatilidade Anualizada (%)", yaxis_title="Retorno Anualizado (%)", template="plotly_dark")
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab3:
        matriz_corr = df_retornos.corr()
        fig_corr = go.Figure(data=go.Heatmap(z=matriz_corr.values, x=matriz_corr.columns, y=matriz_corr.index, colorscale="RdYlBu", zmid=0, text=matriz_corr.round(3).values, texttemplate="%{text}", textfont={"size": 12}))
        fig_corr.update_layout(title="Heatmap de Correlações", template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            at1 = st.selectbox("Primeiro Ativo:", df_retornos.columns, index=0, key="at1_f")
        with col2:
            at2 = st.selectbox("Segundo Ativo:", df_retornos.columns, index=1, key="at2_f")
        if at1 != at2:
            pesos, ret_f, risk_f = fronteira_eficiente_dois_ativos(df_retornos[at1], df_retornos[at2])
            w1, w2 = calcular_carteira_otima(df_retornos[at1], df_retornos[at2])
            ret_ot = w1 * df_retornos[at1].mean() * 252 + w2 * df_retornos[at2].mean() * 252
            vol_ot = np.sqrt(w1**2 * (df_retornos[at1].std() * np.sqrt(252))**2 + w2**2 * (df_retornos[at2].std() * np.sqrt(252))**2 + 2 * w1 * w2 * df_retornos[at1].std() * df_retornos[at2].std() * 252 * df_retornos[at1].corr(df_retornos[at2]))
            fig_fr = go.Figure()
            fig_fr.add_trace(go.Scatter(x=risk_f * 100, y=ret_f * 100, mode="lines", name="Fronteira Eficiente", line=dict(width=3, color="blue")))
            fig_fr.add_trace(go.Scatter(x=[df_retornos[at1].std() * np.sqrt(252) * 100], y=[df_retornos[at1].mean() * 252 * 100], mode="markers+text", text=[at1], textposition="top center", marker=dict(size=12, color="red")))
            fig_fr.add_trace(go.Scatter(x=[df_retornos[at2].std() * np.sqrt(252) * 100], y=[df_retornos[at2].mean() * 252 * 100], mode="markers+text", text=[at2], textposition="top center", marker=dict(size=12, color="green")))
            fig_fr.add_trace(go.Scatter(x=[vol_ot * 100], y=[ret_ot * 100], mode="markers+text", text=["Ótima"], textposition="top center", marker=dict(size=15, color="gold", symbol="star")))
            fig_fr.update_layout(title=f"Fronteira Eficiente: {at1} vs {at2}", xaxis_title="Risco (%)", yaxis_title="Retorno (%)", template="plotly_dark")
            st.plotly_chart(fig_fr, use_container_width=True)

    with tab5:
        at_capm = st.selectbox("Ativo:", df_retornos.columns[1:], key="at_capm")
        rf_capm = st.number_input("Taxa Livre de Risco (% a.a.):", value=10.75, step=0.25) / 100
        if st.button("Calcular CAPM", key="calc_capm"):
            from scipy import stats as sp_stats
            rf_d = rf_capm / 252
            premio_a = df_retornos[at_capm] - rf_d
            premio_m = df_retornos["IBOV"] - rf_d
            beta, alpha, r2, pv, _ = sp_stats.linregress(premio_m, premio_a)
            x_l = np.linspace(premio_m.min(), premio_m.max(), 100)
            fig_c = go.Figure()
            fig_c.add_trace(go.Scatter(x=premio_m * 100, y=premio_a * 100, mode="markers", marker=dict(size=5, opacity=0.5)))
            fig_c.add_trace(go.Scatter(x=x_l * 100, y=(alpha + beta * x_l) * 100, mode="lines", name=f"β={beta:.3f}", line=dict(width=3, color="red")))
            fig_c.update_layout(title=f"CAPM: {at_capm} vs IBOV", xaxis_title="Prêmio IBOV (%)", yaxis_title=f"Prêmio {at_capm} (%)", template="plotly_dark")
            st.plotly_chart(fig_c, use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Beta (β)", f"{beta:.3f}")
            c2.metric("Alpha (α)", f"{alpha*252*100:.2f}% a.a.")
            c3.metric("R²", f"{r2:.3f}")
            c4.metric("P-valor", f"{pv:.4f}")

elif selected_tool == "Estrutura de Capital":
    st.subheader("Análise de Estrutura de Capital")
    tab1, tab2, tab3 = st.tabs(["Calculadora WACC", "Análise CAPM", "Estrutura Ótima"])

    with tab1:
        st.write("**WACC = (E/V × Re) + (D/V × Rd × (1-T))**")
        col1, col2 = st.columns(2)
        with col1:
            pl = st.number_input("Patrimônio Líquido (E) — R$ milhões:", min_value=0.01, value=1000.0, step=10.0)
            div = st.number_input("Dívida Total (D) — R$ milhões:", min_value=0.0, value=500.0, step=10.0)
            re = st.number_input("Custo do Capital Próprio (Re) — %:", min_value=0.0, value=12.0, step=0.1) / 100
            rd = st.number_input("Custo da Dívida (Rd) — %:", min_value=0.0, value=8.0, step=0.1) / 100
        with col2:
            tax = st.number_input("Taxa de Imposto (T) — %:", min_value=0.0, max_value=100.0, value=34.0, step=1.0) / 100
            V = pl + div
            st.write(f"Peso E/V: {pl/V:.1%} | Peso D/V: {div/V:.1%}")
        if st.button("Calcular WACC"):
            wacc = calcular_wacc(pl, div, V, re, rd, tax)
            st.success(f"**WACC = {wacc:.2%}**")
            comp_e = (pl / V) * re
            comp_d = (div / V) * rd * (1 - tax)
            c1, c2, c3 = st.columns(3)
            c1.metric("Componente Equity", f"{comp_e:.2%}")
            c2.metric("Componente Dívida", f"{comp_d:.2%}")
            c3.metric("WACC Total", f"{wacc:.2%}")

    with tab2:
        st.write("**Re = Rf + β × (Rm - Rf)**")
        col1, col2 = st.columns(2)
        with col1:
            rf = st.number_input("Taxa Livre de Risco (Rf) — %:", value=5.0, step=0.1) / 100
            beta = st.number_input("Beta (β):", value=1.2, step=0.1)
            rm = st.number_input("Retorno do Mercado (Rm) — %:", value=12.0, step=0.1) / 100
        with col2:
            re_capm = calcular_capm(rf, beta, rm)
            st.write(f"Prêmio de Risco do Mercado: {(rm-rf):.2%}")
            st.write(f"**Custo do Capital Próprio: {re_capm:.2%}**")

    with tab3:
        st.write("Trade-off entre benefícios fiscais da dívida e custos de dificuldades financeiras.")
        col1, col2 = st.columns(2)
        with col1:
            pl_ot = st.number_input("Patrimônio Líquido Base — R$ milhões:", value=1000.0, step=10.0)
            re_ot = st.number_input("Custo do Capital Próprio — %:", value=12.0, step=0.1) / 100
            rd_ot = st.number_input("Custo da Dívida — %:", value=8.0, step=0.1) / 100
        with col2:
            tax_ot = st.number_input("Taxa de Imposto — %:", value=34.0, step=1.0) / 100
            cf_ot = st.number_input("Taxa de Custo de Falência — %:", value=2.0, step=0.1) / 100
            max_div = st.number_input("Dívida Máxima — R$ milhões:", value=2000.0, step=50.0)
        if st.button("Analisar Estrutura Ótima"):
            df_an = analisar_estrutura_otima(np.linspace(0, max_div, 100), pl_ot, re_ot, rd_ot, tax_ot, cf_ot)
            idx_ot = df_an["WACC"].idxmin()
            ot = df_an.iloc[idx_ot]
            st.metric("WACC Mínimo", f"{ot['WACC']:.2%}", f"Dívida: R$ {ot['Divida']:.0f}M")
            st.metric("Debt Ratio Ótimo", f"{ot['Debt_Ratio']:.1%}")
            fig_ot = go.Figure()
            fig_ot.add_trace(go.Scatter(x=df_an["Debt_Ratio"] * 100, y=df_an["WACC"] * 100, mode="lines", name="WACC (%)", line=dict(color="red", width=3)))
            fig_ot.add_trace(go.Scatter(x=[ot["Debt_Ratio"] * 100], y=[ot["WACC"] * 100], mode="markers", name="Ótimo", marker=dict(size=12, color="gold", symbol="star")))
            fig_ot.update_layout(title="WACC vs Índice de Endividamento", xaxis_title="Debt Ratio (%)", yaxis_title="WACC (%)", template="plotly_dark")
            st.plotly_chart(fig_ot, use_container_width=True)

elif selected_tool == "Dividendos vs JCP":
    st.subheader("Política de Dividendos e Juros sobre Capital Próprio (JCP)")
    col1, col2 = st.columns(2)
    with col1:
        pl_jcp = st.number_input("Patrimônio Líquido (R$ milhões):", min_value=0.01, value=500.0, step=10.0)
        valor_dist = st.number_input("Valor a Distribuir (R$ milhões):", min_value=0.01, value=50.0, step=1.0)
        tjlp = st.number_input("TJLP Anual (%):", min_value=0.0, value=6.0, step=0.1) / 100
        lucro = st.number_input("Lucro Líquido Anual (R$ milhões):", min_value=0.0, value=100.0, step=5.0)
    with col2:
        aliq_ir = st.number_input("Alíquota IR (%):", value=25.0, step=1.0) / 100
        aliq_csll = st.number_input("Alíquota CSLL (%):", value=9.0, step=1.0) / 100
        ir_ret = st.number_input("IR Retido JCP (%):", value=15.0, step=1.0) / 100

    if st.button("Analisar Estratégias"):
        analise = analisar_dividendos_vs_jcp(valor_dist, pl_jcp, tjlp, lucro, aliq_ir, aliq_csll, ir_ret)
        c1, c2, c3 = st.columns(3)
        c1.metric("Limite JCP", f"R$ {analise['limite_jcp']:.2f}M")
        c2.metric("JCP Possível", f"R$ {analise['jcp_possivel']:.2f}M")
        c3.metric("Tax Shield JCP", f"R$ {analise['cenario_jcp']['tax_shield']:.2f}M")
        dados = [
            {"Estratégia": "Dividendos", "Tax Shield": analise["cenario_dividendos"]["tax_shield"], "Custo Empresa": analise["cenario_dividendos"]["custo_empresa"], "Líquido Acionista": analise["cenario_dividendos"]["liquido_acionista"]},
            {"Estratégia": "JCP", "Tax Shield": analise["cenario_jcp"]["tax_shield"], "Custo Empresa": analise["cenario_jcp"]["custo_empresa"], "Líquido Acionista": analise["cenario_jcp"]["liquido_acionista"]},
            {"Estratégia": "Misto", "Tax Shield": analise["cenario_misto"]["tax_shield"], "Custo Empresa": analise["cenario_misto"]["custo_empresa"], "Líquido Acionista": analise["cenario_misto"]["liquido_acionista"]},
        ]
        st.dataframe(pd.DataFrame(dados).round(2), use_container_width=True)
        fig_jcp = go.Figure()
        fig_jcp.add_trace(go.Bar(name="Tax Shield", x=[d["Estratégia"] for d in dados], y=[d["Tax Shield"] for d in dados], marker_color="green"))
        fig_jcp.add_trace(go.Bar(name="Custo Empresa", x=[d["Estratégia"] for d in dados], y=[d["Custo Empresa"] for d in dados], marker_color="red"))
        fig_jcp.update_layout(barmode="group", template="plotly_dark", title="Tax Shield vs Custo Real para a Empresa")
        st.plotly_chart(fig_jcp, use_container_width=True)

elif selected_tool == "Títulos de Renda Fixa":
    st.subheader("Calculadora de Títulos de Renda Fixa")
    tab1, tab2, tab3 = st.tabs(["Cálculo de Preço", "Duration e Convexidade", "Análise de Sensibilidade"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            vf = st.number_input("Valor de Face (R$):", min_value=0.01, value=1000.0, step=10.0)
            cup = st.number_input("Cupom Anual (R$):", min_value=0.0, value=80.0, step=5.0)
            mat = st.number_input("Prazo até Vencimento (anos):", min_value=0.1, value=5.0, step=0.5)
            freq = st.selectbox("Frequência:", [1, 2], format_func=lambda x: "Anual" if x == 1 else "Semestral")
            ytm = st.number_input("YTM (% a.a.):", min_value=0.0, value=9.0, step=0.1) / 100
        with col2:
            if st.button("Calcular Preço", key="calc_preco"):
                preco = calcular_preco_titulo(vf, cup, ytm, mat, freq)
                tipo_bond = "Par Value" if abs(preco - vf) < 1 else ("Discount" if preco < vf else "Premium")
                cor = "blue" if tipo_bond == "Par Value" else ("red" if tipo_bond == "Discount" else "green")
                st.markdown(f'<div style="text-align:center;padding:20px;border:2px solid {cor};border-radius:8px;"><h3 style="color:{cor};">R$ {preco:.2f}</h3><p>{tipo_bond} Bond</p></div>', unsafe_allow_html=True)
                cy = calcular_current_yield(cup, preco) if cup > 0 else 0
                st.write(f"Current Yield: {cy:.2%} | Taxa de Cupom: {(cup/vf):.2%} | YTM: {ytm:.2%}")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            vf_d = st.number_input("Valor de Face (R$):", value=1000.0, step=10.0, key="vf_d")
            cup_d = st.number_input("Cupom Anual (R$):", value=80.0, step=5.0, key="cup_d")
            ytm_d = st.number_input("YTM (% a.a.):", value=9.0, step=0.1, key="ytm_d") / 100
            mat_d = st.number_input("Prazo (anos):", value=5.0, step=0.5, key="mat_d")
            freq_d = st.selectbox("Frequência:", [1, 2], format_func=lambda x: "Anual" if x == 1 else "Semestral", key="freq_d")
        with col2:
            if st.button("Calcular Duration e Convexidade"):
                preco_d = calcular_preco_titulo(vf_d, cup_d, ytm_d, mat_d, freq_d)
                dm = calcular_duration_macaulay(vf_d, cup_d, ytm_d, mat_d, freq_d)
                dmod = calcular_duration_modificada(dm, ytm_d, freq_d)
                conv = calcular_convexidade(vf_d, cup_d, ytm_d, mat_d, freq_d)
                st.metric("Preço", f"R$ {preco_d:.2f}")
                st.metric("Duration Macaulay", f"{dm:.4f} anos")
                st.metric("Duration Modificada", f"{dmod:.4f}")
                st.metric("Convexidade", f"{conv:.6f}")
                st.info(f"Para cada 1% de aumento na taxa, o preço cai aproximadamente {dmod:.2f}%.")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            vf_s = st.number_input("Valor de Face:", value=1000.0, key="vf_s")
            cup_s = st.number_input("Cupom Anual (R$):", value=80.0, key="cup_s")
            mat_s = st.number_input("Prazo (anos):", value=5.0, key="mat_s")
            ytm_b = st.number_input("YTM Base (%):", value=8.0, key="ytm_b") / 100
            rng = st.slider("Range de análise (+/- pp):", 1, 10, 5)
            freq_s = st.selectbox("Frequência:", [1, 2], format_func=lambda x: "Anual" if x == 1 else "Semestral", key="freq_s")
        with col2:
            if st.button("Gerar Análise", key="gerar_sens"):
                taxas = np.linspace(max(0.001, ytm_b - rng / 100), ytm_b + rng / 100, 100)
                precos_s = [calcular_preco_titulo(vf_s, cup_s, t, mat_s, freq_s) for t in taxas]
                preco_b = calcular_preco_titulo(vf_s, cup_s, ytm_b, mat_s, freq_s)
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(x=taxas * 100, y=precos_s, mode="lines", line=dict(width=3, color="blue")))
                fig_s.add_trace(go.Scatter(x=[ytm_b * 100], y=[preco_b], mode="markers", marker=dict(size=12, color="red"), name="YTM Atual"))
                fig_s.add_hline(y=vf_s, line_dash="dash", line_color="green", annotation_text=f"Valor de Face: R$ {vf_s:.0f}")
                fig_s.update_layout(title="Preço vs YTM", xaxis_title="YTM (%)", yaxis_title="Preço (R$)", template="plotly_dark")
                st.plotly_chart(fig_s, use_container_width=True)

elif selected_tool == "Economia do 'e'":
    st.subheader("A Economia do Número 'e' e Crescimento Exponencial")
    tab1, tab2, tab3 = st.tabs(["O Número e", "Capitalização Discreta vs Contínua", "Aplicações"])

    with tab1:
        m_max = st.slider("Número de períodos (m):", min_value=1, max_value=1000000, value=10000, step=1000)
        valores_m = [1, 2, 4, 12, 52, 365, 1000, 10000, m_max]
        aprox_data = [{"m": m, "Valor": f"{calcular_e_aproximacao(m):.8f}", "Diferença de e": f"{abs(calcular_e_aproximacao(m) - np.e):.8f}"} for m in sorted(set(valores_m))]
        st.dataframe(pd.DataFrame(aprox_data), use_container_width=True)
        st.success(f"Valor exato de e: {np.e:.10f}")
        m_vals = np.logspace(0, 6, 100)
        e_ap = [(1 + 1/m)**m for m in m_vals]
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=m_vals, y=e_ap, mode="lines", line=dict(color="blue", width=2)))
        fig_e.add_hline(y=np.e, line_dash="dash", line_color="red", annotation_text=f"e = {np.e:.6f}")
        fig_e.update_layout(title="Convergência de (1 + 1/m)^m para e", xaxis_title="m", yaxis_title="Valor", xaxis_type="log", template="plotly_dark")
        st.plotly_chart(fig_e, use_container_width=True)

    with tab2:
        principal = st.number_input("Capital Inicial ($):", min_value=0.01, value=1000.0, step=100.0)
        taxa_a = st.number_input("Taxa Anual (%):", min_value=0.0, value=10.0, step=0.5) / 100
        tempo_a = st.number_input("Tempo (anos):", min_value=0.1, value=5.0, step=0.5)
        freqs = {"Anual": 1, "Semestral": 2, "Trimestral": 4, "Mensal": 12, "Diária": 365}
        res = [{"Capitalização": n, "Valor Final": f"${valor_futuro_composto(principal, taxa_a, f, tempo_a):,.2f}"} for n, f in freqs.items()]
        res.append({"Capitalização": "Contínua", "Valor Final": f"${valor_futuro_continuo(principal, taxa_a, tempo_a):,.2f}"})
        st.dataframe(pd.DataFrame(res), use_container_width=True)
        t_r = np.linspace(0, tempo_a, 100)
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=t_r, y=[valor_futuro_discreto(principal, taxa_a, t) for t in t_r], mode="lines", name="Anual"))
        fig_c.add_trace(go.Scatter(x=t_r, y=[valor_futuro_composto(principal, taxa_a, 12, t) for t in t_r], mode="lines", name="Mensal"))
        fig_c.add_trace(go.Scatter(x=t_r, y=[valor_futuro_continuo(principal, taxa_a, t) for t in t_r], mode="lines", name="Contínua", line=dict(dash="dash", width=3)))
        fig_c.update_layout(title="Crescimento: Diferentes Frequências de Capitalização", xaxis_title="Tempo (anos)", yaxis_title="Valor ($)", template="plotly_dark")
        st.plotly_chart(fig_c, use_container_width=True)

    with tab3:
        cap_j = st.number_input("Capital ($):", value=10000.0, step=1000.0)
        taxa_j = st.number_input("Taxa (% a.a.):", value=12.0) / 100
        tempo_j = st.slider("Tempo (anos):", 1, 30, 15)
        t_j = np.linspace(0, tempo_j, tempo_j * 4)
        fig_j = go.Figure()
        fig_j.add_trace(go.Scatter(x=t_j, y=[cap_j * (1 + taxa_j * t) for t in t_j], mode="lines", name="Juros Simples", line=dict(color="blue")))
        fig_j.add_trace(go.Scatter(x=t_j, y=[valor_futuro_discreto(cap_j, taxa_j, t) for t in t_j], mode="lines", name="Compostos", line=dict(color="red")))
        fig_j.add_trace(go.Scatter(x=t_j, y=[valor_futuro_continuo(cap_j, taxa_j, t) for t in t_j], mode="lines", name="Contínuos", line=dict(color="green", dash="dash")))
        fig_j.update_layout(title="Juros Simples vs Compostos vs Contínuos", xaxis_title="Tempo (anos)", yaxis_title="Montante ($)", template="plotly_dark")
        st.plotly_chart(fig_j, use_container_width=True)

elif selected_tool == "Hedge Cambial":
    st.subheader("Hedge Cambial")
    st.write("""Este simulador ajuda importadores e exportadores na definição de políticas de hedge cambial
    usando futuros/termo de dólar. Abaixo: 1. Resultado do hedge. 2. Comportamento da taxa de câmbio
    nos últimos dois anos. 3. Mediana das previsões do Boletim Focus-BCB.""")

    tipo_analise = st.selectbox("Tipo de Análise:", ["Exportação", "Importação"], key="hc_tipo")

    col1, col2 = st.columns(2)
    with col1:
        if tipo_analise == "Exportação":
            valor_op = st.number_input("Valor da Exportação (USD):", min_value=0.0, value=100000.0, key="hc_val")
        else:
            valor_op = st.number_input("Valor da Importação (USD):", min_value=0.0, value=100000.0, key="hc_val")
        taxa_cambio_atual = st.number_input("Taxa de Câmbio Atual (USD/BRL):", min_value=0.0, value=5.25, key="hc_spot")
    with col2:
        contrato_futuro = st.number_input("Taxa do Contrato Futuro (USD/BRL):", min_value=0.0, value=5.30, key="hc_fut")
        percentual_hedge = st.slider("Percentual a Ser Hedgeado (%):", min_value=0, max_value=100, value=50, key="hc_pct")

    if valor_op > 0 and taxa_cambio_atual > 0 and contrato_futuro > 0:
        valor_hedgeado = valor_op * (percentual_hedge / 100)
        valor_nao_hedgeado = valor_op - valor_hedgeado

        if tipo_analise == "Exportação":
            resultado_com = valor_hedgeado * contrato_futuro + valor_nao_hedgeado * taxa_cambio_atual
            resultado_sem = valor_op * taxa_cambio_atual
            label_com = f"Recebimento com Hedge ({percentual_hedge}%)"
            label_sem = "Recebimento sem Hedge"
            ylabel = "Recebimento (R$)"
        else:
            resultado_com = valor_hedgeado * contrato_futuro + valor_nao_hedgeado * taxa_cambio_atual
            resultado_sem = valor_op * taxa_cambio_atual
            label_com = f"Custo com Hedge ({percentual_hedge}%)"
            label_sem = "Custo sem Hedge"
            ylabel = "Custo (R$)"

        col1, col2 = st.columns(2)
        col1.metric("Com Hedge", f"R$ {resultado_com:,.2f}")
        col2.metric("Sem Hedge", f"R$ {resultado_sem:,.2f}")

        # Payoff chart
        valores = np.linspace(3.5, 7.0, 100)
        resultado_hedge_var = valor_hedgeado * contrato_futuro + valor_nao_hedgeado * valores
        resultado_sem_var = valor_op * valores

        fig_hc, ax = plt.subplots(figsize=(8, 4))
        ax.plot(valores, resultado_sem_var, label=label_sem)
        ax.plot(valores, resultado_hedge_var, label=label_com, linestyle="--")
        ax.axvline(taxa_cambio_atual, color="red", linewidth=0.8, linestyle="--", label="Taxa Atual")
        ax.set_title("Payoff da Estratégia")
        ax.set_xlabel("Taxa de Câmbio (USD/BRL)")
        ax.set_ylabel(ylabel)
        ax.legend()
        st.pyplot(fig_hc)

    # FX history
    st.subheader("Evolução da Taxa de Câmbio — últimos 2 anos")
    try:
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=2 * 365)).strftime("%Y-%m-%d")
        data_fx = yf.download("BRL=X", start=start_date, end=end_date, progress=False)
        fig_fx, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(data_fx["Close"])
        ax2.set_title("USD/BRL")
        ax2.set_xlabel("Data")
        ax2.set_ylabel("Taxa de Câmbio")
        ax2.tick_params(axis="x", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig_fx)
    except Exception as e:
        st.warning(f"Não foi possível carregar dados de câmbio: {e}")

    # Focus BCB
    st.subheader("Previsões Focus-BCB — Câmbio")
    try:
        from bcb import Expectativas
        em = Expectativas()
        ep = em.get_endpoint("ExpectativasMercadoAnuais")
        ano_ref = datetime.today().year
        previsoes = (ep.query()
                     .filter(ep.Indicador == "Câmbio")
                     .filter(ep.Data >= f"{ano_ref - 1}-01-01")
                     .filter(ep.DataReferencia == ano_ref)
                     .select(ep.Data, ep.Mediana)
                     .orderby(ep.Data.desc())
                     .limit(1000)
                     .collect())
        datas = previsoes["Data"][::-1]
        medianas = previsoes["Mediana"][::-1]
        fig_focus, ax3 = plt.subplots(figsize=(8, 3))
        ax3.plot(datas, medianas)
        ax3.set_title(f"Mediana das Previsões de Câmbio para Final de {ano_ref}")
        ax3.set_xlabel("Data")
        ax3.set_ylabel("Mediana (R$)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_focus)
    except Exception as e:
        st.warning(f"Não foi possível carregar dados do Focus-BCB: {e}")
