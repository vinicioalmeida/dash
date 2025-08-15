import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import pandas as pd

# Configuração da página
st.set_page_config(page_title='Ferramentas Quantitativas')

st.markdown('<span style="color:gold; font-size: 48px">📚</span> <span style="font-size: 48px; font-weight: bold">Ferramentas Quantitativas</span>', unsafe_allow_html=True)
st.markdown("""
    Prof. Vinicio Almeida - https://sites.google.com/view/vinicioalmeida
    """)
st.markdown('---')

# Configuração da barra lateral

selected_tool = st.sidebar.radio(
    "Escolha a Ferramenta:",
    ["Calculadora de VPL", "Estrutura de Capital", "Dividendos vs JCP", "Títulos de Renda Fixa", "Economia do 'e'", "Black-Scholes-Merton", "Gregas de Opções", "Payoff de Opções", "Simulador de Monte Carlo"]
)

st.sidebar.markdown('---')

st.sidebar.markdown("[Simulador cambial](https://simuladorcambio.streamlit.app)")

# Função para Black-Scholes
def black_scholes(S, K, T, r, sigma, tipo='call'):
    d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if tipo.lower() == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Função para simulação Monte Carlo
def monte_carlo_option_pricing(S, K, T, r, sigma, n_sims, n_steps, tipo='call'):
    dt = T / n_steps
    
    # Gerar trajetórias
    np.random.seed(42)  # Para reprodutibilidade
    Z = np.random.standard_normal((n_sims, n_steps))
    
    # Preço inicial
    S_paths = np.zeros((n_sims, n_steps + 1))
    S_paths[:, 0] = S
    
    # Simular trajetórias usando modelo GBM
    for t in range(1, n_steps + 1):
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    # Calcular payoffs
    S_final = S_paths[:, -1]
    if tipo.lower() == 'call':
        payoffs = np.maximum(S_final - K, 0)
    else:
        payoffs = np.maximum(K - S_final, 0)
    
    # Descontar para valor presente
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price, S_paths, S_final, payoffs

# Funções para VPL
def calcular_vpl(investimento_inicial, fluxos_caixa, taxa_desconto):
    """
    Calcula o Valor Presente Líquido (VPL)
    
    Args:
        investimento_inicial: Investimento inicial (C0) - valor negativo
        fluxos_caixa: Lista dos fluxos de caixa futuros [C1, C2, ..., Ct]
        taxa_desconto: Taxa de desconto (r) em decimal
    
    Returns:
        VPL, VP dos fluxos, lista dos VPs individuais
    """
    vp_fluxos = []
    vp_total = 0
    
    for t, fluxo in enumerate(fluxos_caixa, 1):
        vp_fluxo = fluxo / ((1 + taxa_desconto) ** t)
        vp_fluxos.append(vp_fluxo)
        vp_total += vp_fluxo
    
    vpl = investimento_inicial + vp_total  # investimento_inicial já é negativo
    
    return vpl, vp_total, vp_fluxos

def taxa_interna_retorno(investimento_inicial, fluxos_caixa, tentativas=1000):
    """
    Calcula a Taxa Interna de Retorno (TIR) usando busca binária
    """
    def vpl_para_taxa(taxa):
        return calcular_vpl(investimento_inicial, fluxos_caixa, taxa)[0]
    
    # Busca binária para encontrar a TIR
    taxa_min, taxa_max = -0.99, 10.0
    
    for _ in range(tentativas):
        taxa_media = (taxa_min + taxa_max) / 2
        vpl_medio = vpl_para_taxa(taxa_media)
        
        if abs(vpl_medio) < 0.01:  # Precisão
            return taxa_media
        elif vpl_medio > 0:
            taxa_min = taxa_media
        else:
            taxa_max = taxa_media
    
    return taxa_media

# Funções para Títulos de Renda Fixa
def calcular_preco_titulo(valor_face, cupom, ytm, maturidade, freq_cupom=1):
    """
    Calcula o preço de um título de renda fixa
    
    Args:
        valor_face: Valor de face do título
        cupom: Valor do cupom por período
        ytm: Yield to Maturity (taxa ao ano)
        maturidade: Prazo até vencimento (anos)
        freq_cupom: Frequência de pagamento dos cupons por ano (1=anual, 2=semestral)
    
    Returns:
        Preço do título
    """
    if freq_cupom == 1:
        # Cupons anuais
        periodos = int(maturidade)
        taxa_periodo = ytm
        cupom_periodo = cupom
    else:
        # Cupons semestrais
        periodos = int(maturidade * freq_cupom)
        taxa_periodo = ytm / freq_cupom
        cupom_periodo = cupom / freq_cupom
    
    if periodos == 0:
        return valor_face
    
    # Valor presente dos cupons
    vp_cupons = 0
    for t in range(1, periodos + 1):
        vp_cupons += cupom_periodo / ((1 + taxa_periodo) ** t)
    
    # Valor presente do principal
    vp_principal = valor_face / ((1 + taxa_periodo) ** periodos)
    
    return vp_cupons + vp_principal

def calcular_ytm_titulo(preco, valor_face, cupom, maturidade, freq_cupom=1, tentativas=1000):
    """
    Calcula o YTM de um título usando busca binária
    """
    def preco_para_ytm(ytm_teste):
        return calcular_preco_titulo(valor_face, cupom, ytm_teste, maturidade, freq_cupom)
    
    # Busca binária
    ytm_min, ytm_max = 0.001, 1.0
    
    for _ in range(tentativas):
        ytm_medio = (ytm_min + ytm_max) / 2
        preco_calculado = preco_para_ytm(ytm_medio)
        
        if abs(preco_calculado - preco) < 0.01:
            return ytm_medio
        elif preco_calculado > preco:
            ytm_min = ytm_medio
        else:
            ytm_max = ytm_medio
    
    return ytm_medio

def calcular_duration_macaulay(valor_face, cupom, ytm, maturidade, freq_cupom=1):
    """
    Calcula a Duration de Macaulay
    """
    if freq_cupom == 1:
        periodos = int(maturidade)
        taxa_periodo = ytm
        cupom_periodo = cupom
    else:
        periodos = int(maturidade * freq_cupom)
        taxa_periodo = ytm / freq_cupom
        cupom_periodo = cupom / freq_cupom
    
    preco_titulo = calcular_preco_titulo(valor_face, cupom, ytm, maturidade, freq_cupom)
    
    duration_numerador = 0
    
    # Contribuição dos cupons
    for t in range(1, periodos + 1):
        vp_fluxo = cupom_periodo / ((1 + taxa_periodo) ** t)
        duration_numerador += t * vp_fluxo
    
    # Contribuição do principal
    vp_principal = valor_face / ((1 + taxa_periodo) ** periodos)
    duration_numerador += periodos * vp_principal
    
    duration = duration_numerador / preco_titulo
    
    # Ajustar para frequência anual se necessário
    if freq_cupom > 1:
        duration = duration / freq_cupom
    
    return duration

def calcular_duration_modificada(duration_macaulay, ytm, freq_cupom=1):
    """
    Calcula a Duration Modificada
    """
    return duration_macaulay / (1 + ytm/freq_cupom)

def calcular_convexidade(valor_face, cupom, ytm, maturidade, freq_cupom=1):
    """
    Calcula a convexidade de um título
    """
    if freq_cupom == 1:
        periodos = int(maturidade)
        taxa_periodo = ytm
        cupom_periodo = cupom
    else:
        periodos = int(maturidade * freq_cupom)
        taxa_periodo = ytm / freq_cupom
        cupom_periodo = cupom / freq_cupom
    
    preco_titulo = calcular_preco_titulo(valor_face, cupom, ytm, maturidade, freq_cupom)
    
    convexidade_numerador = 0
    
    # Contribuição dos cupons
    for t in range(1, periodos + 1):
        vp_fluxo = cupom_periodo / ((1 + taxa_periodo) ** t)
        convexidade_numerador += t * (t + 1) * vp_fluxo
    
    # Contribuição do principal
    vp_principal = valor_face / ((1 + taxa_periodo) ** periodos)
    convexidade_numerador += periodos * (periodos + 1) * vp_principal
    
    convexidade = convexidade_numerador / (preco_titulo * ((1 + taxa_periodo) ** 2))
    
    # Ajustar para frequência anual se necessário
    if freq_cupom > 1:
        convexidade = convexidade / (freq_cupom ** 2)
    
    return convexidade

def calcular_current_yield(cupom_anual, preco_mercado):
    """
    Calcula o Current Yield
    """
    return cupom_anual / preco_mercado

def estimar_variacao_preco(duration_modificada, convexidade, variacao_ytm):
    """
    Estima a variação percentual no preço usando Duration e Convexidade
    """
    variacao_duration = -duration_modificada * variacao_ytm
    variacao_convexidade = 0.5 * convexidade * (variacao_ytm ** 2)
    
    return variacao_duration + variacao_convexidade

# Funções para Estrutura de Capital
def calcular_capm(rf, beta, rm):
    """Calcula o custo do capital próprio usando CAPM"""
    return rf + beta * (rm - rf)

def calcular_custo_divida_apos_imposto(kd, tax_rate):
    """Calcula o custo da dívida após impostos"""
    return kd * (1 - tax_rate)

def calcular_wacc(E, D, V, Re, Rd, tax_rate):
    """Calcula o WACC (Weighted Average Cost of Capital)"""
    return (E/V) * Re + (D/V) * Rd * (1 - tax_rate)

def analisar_estrutura_otima(valores_divida, patrimonio_liquido, custo_equity, custo_divida, tax_rate, custo_falencia_rate=0.02):
    """Analisa a estrutura de capital ótima considerando benefícios fiscais e custos de falência"""
    resultados = []
    
    for D in valores_divida:
        V = D + patrimonio_liquido
        debt_ratio = D / V
        
        # Benefício fiscal
        tax_shield = tax_rate * D
        
        # Custo de falência (simplificado)
        custo_falencia = custo_falencia_rate * debt_ratio**2 * V
        
        # WACC
        wacc = calcular_wacc(patrimonio_liquido, D, V, custo_equity, custo_divida, tax_rate)
        
        # Valor da empresa (simplificado)
        valor_empresa = V + tax_shield - custo_falencia
        
        resultados.append({
            'Divida': D,
            'Patrimonio_Liquido': patrimonio_liquido,
            'Valor_Total': V,
            'Debt_Ratio': debt_ratio,
            'Equity_Ratio': patrimonio_liquido/V,
            'Tax_Shield': tax_shield,
            'Custo_Falencia': custo_falencia,
            'WACC': wacc,
            'Valor_Empresa': valor_empresa
        })
    
    return pd.DataFrame(resultados)

# Funções para Dividendos vs JCP
def calcular_limite_jcp(patrimonio_liquido, tjlp_anual):
    return patrimonio_liquido * tjlp_anual

def calcular_tax_shield_jcp(valor_jcp, aliquota_ir=0.25, aliquota_csll=0.09):
    aliquota_total = aliquota_ir + aliquota_csll
    return valor_jcp * aliquota_total

def calcular_recebimento_liquido_jcp(valor_jcp, ir_retido=0.15):
    return valor_jcp * (1 - ir_retido)

def analisar_dividendos_vs_jcp(valor_distribuicao, patrimonio_liquido, tjlp_anual, lucro_liquido, 
                              aliquota_ir=0.25, aliquota_csll=0.09, ir_retido_jcp=0.15):
    limite_jcp = calcular_limite_jcp(patrimonio_liquido, tjlp_anual)
    
    # Cenário 1: Dividendos
    dividendos = valor_distribuicao
    tax_shield_div = 0
    liquido_acionista_div = dividendos
    custo_empresa_div = dividendos
    
    # Cenário 2: JCP
    jcp_possivel = min(valor_distribuicao, limite_jcp)
    tax_shield_jcp = calcular_tax_shield_jcp(jcp_possivel, aliquota_ir, aliquota_csll)
    liquido_acionista_jcp = calcular_recebimento_liquido_jcp(jcp_possivel, ir_retido_jcp)
    custo_empresa_jcp = jcp_possivel - tax_shield_jcp
    
    # Cenário 3: Misto
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
        'valor_distribuicao': valor_distribuicao,
        'limite_jcp': limite_jcp,
        'jcp_possivel': jcp_possivel,
        'dividendo_complementar': dividendo_complementar,
        'cenario_dividendos': {
            'valor': dividendos,
            'tax_shield': tax_shield_div,
            'liquido_acionista': liquido_acionista_div,
            'custo_empresa': custo_empresa_div
        },
        'cenario_jcp': {
            'valor': jcp_possivel,
            'tax_shield': tax_shield_jcp,
            'liquido_acionista': liquido_acionista_jcp,
            'custo_empresa': custo_empresa_jcp
        },
        'cenario_misto': {
            'jcp': jcp_possivel,
            'dividendos': dividendo_complementar,
            'tax_shield': tax_shield_misto,
            'liquido_acionista': liquido_acionista_misto,
            'custo_empresa': custo_empresa_misto
        }
    }

# Funções para Economia do e e Crescimento
def calcular_e_aproximacao(m):
    """Calcula aproximação de e usando (1 + 1/m)^m"""
    return (1 + 1/m) ** m

def valor_futuro_discreto(principal, taxa, tempo):
    """Calcula valor futuro com capitalização discreta"""
    return principal * (1 + taxa) ** tempo

def valor_futuro_continuo(principal, taxa, tempo):
    """Calcula valor futuro com capitalização contínua"""
    return principal * np.exp(taxa * tempo)

def valor_futuro_composto(principal, taxa_nominal, freq_capitalizacao, tempo):
    """Calcula valor futuro com capitalização composta"""
    return principal * (1 + taxa_nominal/freq_capitalizacao) ** (freq_capitalizacao * tempo)

def taxa_discreta_para_continua(taxa_discreta):
    """Converte taxa discreta para contínua: r = ln(1 + i)"""
    return np.log(1 + taxa_discreta)

def taxa_continua_para_discreta(taxa_continua):
    """Converte taxa contínua para discreta: i = e^r - 1"""
    return np.exp(taxa_continua) - 1

def valor_presente_discreto(valor_futuro, taxa, tempo):
    """Calcula valor presente com desconto discreto"""
    return valor_futuro / (1 + taxa) ** tempo

def valor_presente_continuo(valor_futuro, taxa, tempo):
    """Calcula valor presente com desconto contínuo"""
    return valor_futuro * np.exp(-taxa * tempo)

def crescimento_populacional(populacao_inicial, taxa_crescimento, tempo):
    """Modelo de crescimento populacional exponencial"""
    return populacao_inicial * np.exp(taxa_crescimento * tempo)

# Implementação das funcionalidades
if selected_tool == "Calculadora de VPL":
    st.subheader('Calculadora de Valor Presente Líquido (VPL)')
    
    st.write("""
    O **Valor Presente Líquido (VPL)** é uma ferramenta importante para avaliação de projetos de investimento.
    
    **Equação:** VPL = C₀ + Σ[Cₜ/(1+r)ᵗ]
    
    Onde:
    - C₀ = Investimento inicial (fluxo negativo)
    - Cₜ = Fluxo de caixa no período t
    - r = Taxa de desconto (custo de capital)
    - t = Período
    """)
    
    # Tabs para diferentes tipos de análise
    tab1, tab2, tab3 = st.tabs(["Cálculo Básico", "Análise de Sensibilidade", "Múltiplos Projetos"])
    
    with tab1:
        st.subheader("Cálculo do VPL")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dados do Projeto:**")
            investimento_inicial = st.number_input(
                "Investimento Inicial (R$ mil):", 
                value=100.0, 
                step=10.0,
                help="Valor investido no início do projeto (será considerado negativo)"
            )
            
            taxa_desconto = st.number_input(
                "Taxa de Desconto (% a.a.):", 
                min_value=0.0, 
                max_value=50.0, 
                value=10.0, 
                step=0.5
            ) / 100
            
            # Número de períodos
            num_periodos = st.slider("Número de Períodos:", min_value=1, max_value=10, value=5)
        
        with col2:
            st.write("**Fluxos de Caixa Futuros (R$ mil):**")
            fluxos_caixa = []
            
            for i in range(num_periodos):
                fluxo = st.number_input(
                    f"Ano {i+1}:", 
                    value=30.0, 
                    step=5.0, 
                    key=f"fluxo_{i}"
                )
                fluxos_caixa.append(fluxo)
        
        if st.button("Calcular VPL", key="calc_vpl_basic"):
            # Cálculos
            investimento_negativo = -abs(investimento_inicial)
            vpl, vp_total, vp_fluxos = calcular_vpl(investimento_negativo, fluxos_caixa, taxa_desconto)
            
            # Calcular TIR
            try:
                tir = taxa_interna_retorno(investimento_negativo, fluxos_caixa)
                tir_percent = tir * 100
            except:
                tir_percent = "N/A"
            
            # Resultados principais
            st.subheader("Resultados da Análise")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cor_vpl = "green" if vpl > 0 else "red"
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid {cor_vpl}; border-radius: 10px;">
                    <h3 style="color: {cor_vpl};">VPL</h3>
                    <h2 style="color: {cor_vpl};">R$ {vpl:.2f} mil</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("VP dos Fluxos", f"R$ {vp_total:.2f} mil")
                st.metric("Investimento Inicial", f"R$ {investimento_inicial:.2f} mil")
            
            with col3:
                if isinstance(tir_percent, (int, float)):
                    cor_tir = "green" if tir_percent > taxa_desconto * 100 else "red"
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px;">
                        <h4>TIR</h4>
                        <h3 style="color: {cor_tir};">{tir_percent:.2f}%</h3>
                        <small>Taxa de Desconto: {taxa_desconto*100:.1f}%</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.metric("TIR", "N/A")
            
            # Interpretação
            st.subheader("Interpretação do Resultado")
            
            if vpl > 0:
                st.success(f"""
                PROJETO VIÁVEL
                
                O VPL positivo de R$ {vpl:.2f} mil indica que o projeto:
                - Gera valor para a empresa
                - Tem retorno superior ao custo de capital ({taxa_desconto*100:.1f}%)
                - Deve ser **ACEITO**
                """)
            elif vpl < 0:
                st.error(f"""
                PROJETO NÃO VIÁVEL
                
                O VPL negativo de R$ {vpl:.2f} mil indica que o projeto:
                - Destrói valor para a empresa
                - Tem retorno inferior ao custo de capital ({taxa_desconto*100:.1f}%)
                - Deve ser **REJEITADO**
                """)
            else:
                st.warning("**PROJETO NEUTRO** - VPL = 0. O retorno é exatamente igual ao custo de capital.")
            
            # Tabela detalhada
            st.subheader("Detalhamento dos Cálculos")
            
            dados_tabela = []
            dados_tabela.append({
                'Período': 0,
                'Fluxo de Caixa': f"R$ {investimento_negativo:.2f}",
                'Fator de Desconto': "1,0000",
                'Valor Presente': f"R$ {investimento_negativo:.2f}"
            })
            
            for i, (fluxo, vp) in enumerate(zip(fluxos_caixa, vp_fluxos), 1):
                fator_desconto = 1 / ((1 + taxa_desconto) ** i)
                dados_tabela.append({
                    'Período': i,
                    'Fluxo de Caixa': f"R$ {fluxo:.2f}",
                    'Fator de Desconto': f"{fator_desconto:.4f}",
                    'Valor Presente': f"R$ {vp:.2f}"
                })
            
            df_detalhes = pd.DataFrame(dados_tabela)
            st.dataframe(df_detalhes, use_container_width=True)
            
            # Gráfico dos fluxos de caixa
            st.subheader("Visualização dos Fluxos de Caixa")
            
            periodos = list(range(num_periodos + 1))
            fluxos_totais = [investimento_negativo] + fluxos_caixa
            vp_totais = [investimento_negativo] + vp_fluxos
            
            fig = go.Figure()
            
            # Fluxos nominais
            fig.add_trace(go.Bar(
                x=periodos,
                y=fluxos_totais,
                name='Fluxos Nominais',
                marker_color=['red' if x < 0 else 'lightblue' for x in fluxos_totais],
                opacity=0.7
            ))
            
            # Valores presentes
            fig.add_trace(go.Bar(
                x=periodos,
                y=vp_totais,
                name='Valores Presentes',
                marker_color=['darkred' if x < 0 else 'darkblue' for x in vp_totais],
                opacity=0.9
            ))
            
            fig.update_layout(
                title="Fluxos de Caixa: Nominais vs Valores Presentes",
                xaxis_title="Período",
                yaxis_title="Valor (R$ mil)",
                template="plotly_dark",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Análise de Sensibilidade")
        st.write("Analise como mudanças na taxa de desconto afetam o VPL do projeto.")
        
        if 'fluxos_caixa' in locals() and 'investimento_inicial' in locals():
            # Range de taxas para análise
            taxa_min = st.number_input("Taxa Mínima (%)", value=5.0, step=1.0) / 100
            taxa_max = st.number_input("Taxa Máxima (%)", value=20.0, step=1.0) / 100
            
            if st.button("Gerar Análise de Sensibilidade"):
                taxas = np.linspace(taxa_min, taxa_max, 50)
                vpls = []
                
                for taxa in taxas:
                    vpl_temp, _, _ = calcular_vpl(-investimento_inicial, fluxos_caixa, taxa)
                    vpls.append(vpl_temp)
                
                # Gráfico de sensibilidade
                fig_sens = go.Figure()
                
                fig_sens.add_trace(go.Scatter(
                    x=taxas * 100,
                    y=vpls,
                    mode='lines',
                    name='VPL',
                    line=dict(width=3, color='blue')
                ))
                
                # Linha do VPL = 0
                fig_sens.add_hline(y=0, line_dash="dash", line_color="red", 
                                   annotation_text="VPL = 0")
                
                # Marcar taxa atual
                vpl_atual, _, _ = calcular_vpl(-investimento_inicial, fluxos_caixa, taxa_desconto)
                fig_sens.add_trace(go.Scatter(
                    x=[taxa_desconto * 100],
                    y=[vpl_atual],
                    mode='markers',
                    name=f'Taxa Atual ({taxa_desconto*100:.1f}%)',
                    marker=dict(size=10, color='red')
                ))
                
                fig_sens.update_layout(
                    title="Análise de Sensibilidade: VPL vs Taxa de Desconto",
                    xaxis_title="Taxa de Desconto (%)",
                    yaxis_title="VPL (R$ mil)",
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig_sens, use_container_width=True)
                
                # Encontrar TIR (onde VPL = 0)
                vpl_zero_idx = np.argmin(np.abs(vpls))
                tir_aprox = taxas[vpl_zero_idx] * 100
                
                st.info(f"""
                **Insights da Análise de Sensibilidade:**
                
                - **TIR Aproximada:** {tir_aprox:.2f}%
                - **Sensibilidade:** {'Alta' if abs(vpls[0] - vpls[-1]) > 100 else 'Baixa'}
                - **Ponto de Equilíbrio:** Taxa de desconto ≈ {tir_aprox:.1f}%
                """)
        else:
            st.warning("Execute primeiro o cálculo básico do VPL na aba anterior.")
    
    with tab3:
        st.subheader("Comparação de Múltiplos Projetos")
        st.write("Compare o VPL de diferentes projetos para apoiar a decisão de investimento.")
        
        num_projetos = st.slider("Número de Projetos a Comparar:", 2, 5, 3)
        taxa_comparacao = st.number_input("Taxa de Desconto para Comparação (%):", value=10.0) / 100
        
        projetos_dados = []
        
        for i in range(num_projetos):
            st.write(f"**Projeto {chr(65+i)}:**")
            col1, col2 = st.columns(2)
            
            with col1:
                inv_inicial = st.number_input(f"Investimento Inicial (R$ mil):", value=100.0, key=f"inv_{i}")
                periodos = st.slider(f"Períodos:", 1, 8, 5, key=f"per_{i}")
            
            with col2:
                fluxos = []
                for j in range(periodos):
                    fluxo = st.number_input(f"Ano {j+1}:", value=30.0, key=f"fluxo_{i}_{j}")
                    fluxos.append(fluxo)
            
            projetos_dados.append({
                'nome': f'Projeto {chr(65+i)}',
                'investimento': inv_inicial,
                'fluxos': fluxos
            })
        
        if st.button("Comparar Projetos"):
            resultados_projetos = []
            
            for projeto in projetos_dados:
                vpl, vp_total, _ = calcular_vpl(-projeto['investimento'], projeto['fluxos'], taxa_comparacao)
                
                try:
                    tir = taxa_interna_retorno(-projeto['investimento'], projeto['fluxos']) * 100
                except:
                    tir = "N/A"
                
                il = vp_total / projeto['investimento'] if projeto['investimento'] > 0 else 0  # Índice de Lucratividade
                
                resultados_projetos.append({
                    'Projeto': projeto['nome'],
                    'Investimento (R$ mil)': projeto['investimento'],
                    'VPL (R$ mil)': vpl,
                    'TIR (%)': f"{tir:.2f}" if isinstance(tir, (int, float)) else tir,
                    'IL': f"{il:.2f}",
                    'Decisão': 'ACEITAR' if vpl > 0 else 'REJEITAR'
                })
            
            df_resultados = pd.DataFrame(resultados_projetos)
            
            # Colorir as linhas baseado na decisão
            st.subheader("Resumo Comparativo")
            st.dataframe(df_resultados, use_container_width=True)
            
            # Gráfico comparativo
            fig_comp = go.Figure()
            
            projetos_nomes = [r['Projeto'] for r in resultados_projetos]
            vpls = [r['VPL (R$ mil)'] for r in resultados_projetos]
            cores = ['green' if vpl > 0 else 'red' for vpl in vpls]
            
            fig_comp.add_trace(go.Bar(
                x=projetos_nomes,
                y=vpls,
                marker_color=cores,
                text=[f'R$ {vpl:.1f}' for vpl in vpls],
                textposition='auto'
            ))
            
            fig_comp.add_hline(y=0, line_dash="dash", line_color="white")
            
            fig_comp.update_layout(
                title="Comparação de VPL entre Projetos",
                xaxis_title="Projetos",
                yaxis_title="VPL (R$ mil)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Recomendação
            projeto_melhor = max(resultados_projetos, key=lambda x: x['VPL (R$ mil)'])
            if projeto_melhor['VPL (R$ mil)'] > 0:
                st.success(f"""
                **Recomendação:** {projeto_melhor['Projeto']}
                
                - Maior VPL: R$ {projeto_melhor['VPL (R$ mil)']:.2f} mil
                - TIR: {projeto_melhor['TIR (%)']}%
                - Índice de Lucratividade: {projeto_melhor['IL']}
                """)
            else:
                st.warning("Nenhum projeto apresenta VPL positivo. Considere revisar os parâmetros ou buscar alternativas.")
    
    # Seção educativa
    with st.expander("Conceitos Importantes sobre VPL"):
        st.markdown("""
        ### O que é VPL?
        
        O **Valor Presente Líquido** é a diferença entre o valor presente dos fluxos de caixa futuros e o investimento inicial.
        
        ### Regra de Decisão:
        - **VPL > 0:** ACEITAR o projeto (cria valor)
        - **VPL < 0:** REJEITAR o projeto (destrói valor)  
        - **VPL = 0:** INDIFERENTE (retorno = custo de capital)
        
        ### Vantagens do VPL:
        - Considera o valor do dinheiro no tempo
        - Usa todos os fluxos de caixa do projeto
        - Permite comparação direta entre projetos
        - Indica o valor criado em termos absolutos
        
        ### Taxa Interna de Retorno (TIR):
        - Taxa que torna o VPL = 0
        - Se TIR > taxa de desconto → projeto viável
        - Se TIR < taxa de desconto → projeto não viável
        
        ### Índice de Lucratividade (IL):
        - IL = VP dos fluxos / Investimento inicial
        - IL > 1 → projeto viável
        - Útil para comparar projetos com investimentos diferentes
        """)

elif selected_tool == "Títulos de Renda Fixa":
    st.subheader('Calculadora de Títulos de Renda Fixa')
    
    st.write("""
    Esta ferramenta permite calcular preços, yields, duration e convexidade de títulos de renda fixa, 
    incluindo exemplos dos títulos do **Tesouro Direto**.
    """)
    
    # Tabs para diferentes análises
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Cálculo de Preço", 
        "Cálculo de YTM", 
        "Duration e Convexidade", 
        "Análise de Sensibilidade", 
        "Comparação de Títulos"
    ])
    
    with tab1:
        st.subheader("Cálculo do Preço do Título")
        st.write("**Equação:** Preço = Σ[Cupom/(1+YTM)^t] + ValorFace/(1+YTM)^n")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parâmetros do Título:**")
            
            # Seleção de tipo de título
            tipo_titulo = st.selectbox(
                "Tipo de Título (exemplos Tesouro Direto):",
                [
                    "Personalizado",
                    "Tesouro Prefixado (LTN)",
                    "Tesouro IPCA+ Principal (NTN-B Principal)", 
                    "Tesouro IPCA+ com Juros (NTN-B)",
                    "Tesouro Selic (LFT)"
                ]
            )
            
            # Valores padrão baseados no tipo selecionado
            if tipo_titulo == "Tesouro Prefixado (LTN)":
                valor_face_default = 1000.0
                cupom_default = 0.0
                maturidade_default = 3.0
                freq_default = 1
                ytm_default = 10.5
            elif tipo_titulo == "Tesouro IPCA+ Principal (NTN-B Principal)":
                valor_face_default = 1000.0
                cupom_default = 0.0
                maturidade_default = 5.0
                freq_default = 1
                ytm_default = 5.8
            elif tipo_titulo == "Tesouro IPCA+ com Juros (NTN-B)":
                valor_face_default = 1000.0
                cupom_default = 60.0  # 6% a.a.
                maturidade_default = 10.0
                freq_default = 2
                ytm_default = 5.8
            elif tipo_titulo == "Tesouro Selic (LFT)":
                valor_face_default = 1000.0
                cupom_default = 0.0
                maturidade_default = 2.0
                freq_default = 1
                ytm_default = 11.75
            else:
                valor_face_default = 1000.0
                cupom_default = 80.0
                maturidade_default = 5.0
                freq_default = 1
                ytm_default = 8.0
            
            valor_face = st.number_input(
                "Valor de Face (R$):", 
                min_value=0.01, 
                value=valor_face_default, 
                step=10.0
            )
            
            cupom_anual = st.number_input(
                "Cupom Anual (R$):", 
                min_value=0.0, 
                value=cupom_default, 
                step=5.0,
                help="Para títulos zero-cupom, deixe em 0"
            )
            
            maturidade = st.number_input(
                "Prazo até Vencimento (anos):", 
                min_value=0.1, 
                value=maturidade_default, 
                step=0.5
            )
            
            freq_cupom = st.selectbox(
                "Frequência de Pagamento:",
                [1, 2],
                index=0 if freq_default == 1 else 1,
                format_func=lambda x: "Anual" if x == 1 else "Semestral"
            )
            
            ytm = st.number_input(
                "YTM - Yield to Maturity (% a.a.):", 
                min_value=0.0, 
                value=ytm_default, 
                step=0.1
            ) / 100
        
        with col2:
            if st.button("Calcular Preço do Título", key="calc_preco"):
                preco_calculado = calcular_preco_titulo(valor_face, cupom_anual, ytm, maturidade, freq_cupom)
                
                # Classificação do título
                taxa_cupom = cupom_anual / valor_face if valor_face > 0 else 0
                
                if abs(preco_calculado - valor_face) < 1:
                    tipo_bond = "Par Value Bond"
                    cor_preco = "blue"
                elif preco_calculado < valor_face:
                    tipo_bond = "Discount Bond"
                    cor_preco = "red"
                else:
                    tipo_bond = "Premium Bond"
                    cor_preco = "green"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid {cor_preco}; border-radius: 10px;">
                    <h3 style="color: {cor_preco};">Preço do Título</h3>
                    <h2 style="color: {cor_preco};">R$ {preco_calculado:.2f}</h2>
                    <p><strong>{tipo_bond}</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detalhes do cálculo
                st.subheader("Análise Detalhada")
                
                col1_det, col2_det = st.columns(2)
                
                with col1_det:
                    current_yield = calcular_current_yield(cupom_anual, preco_calculado) if cupom_anual > 0 else 0
                    
                    st.write("**Métricas do Título:**")
                    st.write(f"• Taxa de Cupom: {taxa_cupom:.2%}")
                    st.write(f"• YTM: {ytm:.2%}")
                    st.write(f"• Current Yield: {current_yield:.2%}")
                    st.write(f"• Prêmio/Desconto: R$ {preco_calculado - valor_face:.2f}")
                    st.write(f"• Prêmio/Desconto (%): {((preco_calculado - valor_face)/valor_face)*100:.2f}%")
                
                with col2_det:
                    st.write("**Interpretação:**")
                    if tipo_bond == "Par Value Bond":
                        st.info("Título negociado ao par: Taxa de cupom = YTM")
                    elif tipo_bond == "Discount Bond":
                        st.warning("Título com desconto: Taxa de cupom < YTM")
                    else:
                        st.success("Título com prêmio: Taxa de cupom > YTM")
                    
                    if cupom_anual == 0:
                        st.info("Título zero-cupom: retorno apenas na maturidade")
                
                # Fluxo de caixa
                st.subheader("Fluxo de Caixa do Título")
                
                periodos = int(maturidade * freq_cupom)
                taxa_periodo = ytm / freq_cupom
                cupom_periodo = cupom_anual / freq_cupom if freq_cupom > 1 else cupom_anual
                
                fluxos_data = []
                for t in range(0, periodos + 1):
                    if t == 0:
                        fluxo = -preco_calculado
                        vp = fluxo
                    elif t < periodos:
                        fluxo = cupom_periodo
                        vp = fluxo / ((1 + taxa_periodo) ** t)
                    else:
                        fluxo = cupom_periodo + valor_face
                        vp = fluxo / ((1 + taxa_periodo) ** t)
                    
                    fluxos_data.append({
                        'Período': t,
                        'Fluxo de Caixa': f"R$ {fluxo:.2f}",
                        'Valor Presente': f"R$ {vp:.2f}"
                    })
                
                df_fluxos = pd.DataFrame(fluxos_data)
                st.dataframe(df_fluxos, use_container_width=True)
                
                # Gráfico dos fluxos
                periodos_plot = list(range(periodos + 1))
                fluxos_plot = [-preco_calculado] + [cupom_periodo] * (periodos - 1) + [cupom_periodo + valor_face]
                
                fig_fluxos = go.Figure()
                
                fig_fluxos.add_trace(go.Bar(
                    x=periodos_plot,
                    y=fluxos_plot,
                    marker_color=['red'] + ['lightblue'] * (periodos - 1) + ['darkblue'],
                    text=[f'R$ {f:.0f}' for f in fluxos_plot],
                    textposition='auto'
                ))
                
                fig_fluxos.update_layout(
                    title="Fluxo de Caixa do Título",
                    xaxis_title="Período",
                    yaxis_title="Fluxo de Caixa (R$)",
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig_fluxos, use_container_width=True)
    
    with tab2:
        st.subheader("Cálculo do YTM (Yield to Maturity)")
        st.write("Encontra a taxa que iguala o preço de mercado ao valor presente dos fluxos futuros.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dados do Título:**")
            
            preco_mercado = st.number_input(
                "Preço de Mercado (R$):", 
                min_value=0.01, 
                value=885.0, 
                step=1.0,
                key="preco_ytm"
            )
            
            valor_face_ytm = st.number_input(
                "Valor de Face (R$):", 
                min_value=0.01, 
                value=1000.0, 
                step=10.0,
                key="vf_ytm"
            )
            
            cupom_anual_ytm = st.number_input(
                "Cupom Anual (R$):", 
                min_value=0.0, 
                value=80.0, 
                step=5.0,
                key="cupom_ytm"
            )
            
            maturidade_ytm = st.number_input(
                "Prazo até Vencimento (anos):", 
                min_value=0.1, 
                value=9.0, 
                step=0.5,
                key="mat_ytm"
            )
            
            freq_cupom_ytm = st.selectbox(
                "Frequência de Pagamento:",
                [1, 2],
                format_func=lambda x: "Anual" if x == 1 else "Semestral",
                key="freq_ytm"
            )
        
        with col2:
            if st.button("Calcular YTM", key="calc_ytm"):
                ytm_calculado = calcular_ytm_titulo(
                    preco_mercado, valor_face_ytm, cupom_anual_ytm, maturidade_ytm, freq_cupom_ytm
                )
                
                st.success(f"**YTM = {ytm_calculado:.4f} ou {ytm_calculado*100:.2f}% a.a.**")
                
                # Verificação
                preco_verificacao = calcular_preco_titulo(
                    valor_face_ytm, cupom_anual_ytm, ytm_calculado, maturidade_ytm, freq_cupom_ytm
                )
                
                st.write("**Verificação:**")
                st.write(f"• Preço com YTM calculado: R$ {preco_verificacao:.2f}")
                st.write(f"• Preço de mercado: R$ {preco_mercado:.2f}")
                st.write(f"• Diferença: R$ {abs(preco_verificacao - preco_mercado):.2f}")
                
                # Comparação com outras métricas
                current_yield_ytm = calcular_current_yield(cupom_anual_ytm, preco_mercado) if cupom_anual_ytm > 0 else 0
                
                st.subheader("Comparação de Métricas de Rendimento")
                
                metricas_comp = pd.DataFrame({
                    'Métrica': ['YTM', 'Current Yield', 'Taxa de Cupom'],
                    'Taxa (%)': [
                        f"{ytm_calculado*100:.2f}%",
                        f"{current_yield_ytm*100:.2f}%",
                        f"{(cupom_anual_ytm/valor_face_ytm)*100:.2f}%"
                    ],
                    'Descrição': [
                        'Taxa de retorno até vencimento',
                        'Rendimento atual (cupom/preço)',
                        'Cupom/valor de face'
                    ]
                })
                
                st.table(metricas_comp)
    
    with tab3:
        st.subheader("Duration e Convexidade")
        st.write("**Duration:** Medida de sensibilidade do preço às mudanças nas taxas de juros")
        st.write("**Convexidade:** Medida de segunda ordem da relação preço-taxa")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parâmetros do Título:**")
            
            valor_face_dur = st.number_input(
                "Valor de Face (R$):", 
                min_value=0.01, 
                value=1000.0, 
                step=10.0,
                key="vf_dur"
            )
            
            cupom_anual_dur = st.number_input(
                "Cupom Anual (R$):", 
                min_value=0.0, 
                value=80.0, 
                step=5.0,
                key="cupom_dur"
            )
            
            ytm_dur = st.number_input(
                "YTM (% a.a.):", 
                min_value=0.0, 
                value=9.0, 
                step=0.1,
                key="ytm_dur"
            ) / 100
            
            maturidade_dur = st.number_input(
                "Prazo até Vencimento (anos):", 
                min_value=0.1, 
                value=5.0, 
                step=0.5,
                key="mat_dur"
            )
            
            freq_cupom_dur = st.selectbox(
                "Frequência de Pagamento:",
                [1, 2],
                format_func=lambda x: "Anual" if x == 1 else "Semestral",
                key="freq_dur"
            )
        
        with col2:
            if st.button("Calcular Duration e Convexidade", key="calc_duration"):
                # Cálculos
                preco_dur = calcular_preco_titulo(valor_face_dur, cupom_anual_dur, ytm_dur, maturidade_dur, freq_cupom_dur)
                duration_mac = calcular_duration_macaulay(valor_face_dur, cupom_anual_dur, ytm_dur, maturidade_dur, freq_cupom_dur)
                duration_mod = calcular_duration_modificada(duration_mac, ytm_dur, freq_cupom_dur)
                convexidade = calcular_convexidade(valor_face_dur, cupom_anual_dur, ytm_dur, maturidade_dur, freq_cupom_dur)
                
                # Resultados
                st.write("**Resultados:**")
                st.write(f"• Preço atual: R$ {preco_dur:.2f}")
                st.write(f"• Duration de Macaulay: {duration_mac:.4f} anos")
                st.write(f"• Duration Modificada: {duration_mod:.4f}")
                st.write(f"• Convexidade: {convexidade:.6f}")
                
                # Interpretação
                st.subheader("Interpretação dos Resultados")
                
                col1_int, col2_int = st.columns(2)
                
                with col1_int:
                    st.write("**Duration Modificada:**")
                    st.write(f"Para cada 1% de aumento na taxa:")
                    st.write(f"• Preço cai aproximadamente {duration_mod:.2f}%")
                    st.write(f"• Valor absoluto: R$ {preco_dur * duration_mod * 0.01:.2f}")
                
                with col2_int:
                    st.write("**Convexidade:**")
                    if convexidade > 0:
                        st.write("• Proteção contra aumentos de taxa")
                        st.write("• Ganhos maiores em quedas de taxa")
                        st.write("• Característica desejável")
                    else:
                        st.write("• Convexidade negativa")
                        st.write("• Maior risco de taxa")
        
        # Agora colocar toda a seção de estimativa fora dos botões mas dentro da aba
        st.markdown("---")
        st.subheader("Simulador de Variação de Preço")
        st.write("Use os parâmetros acima para simular como mudanças nas taxas afetam o preço:")
        
        # Parâmetros para simulação
        col1_sim, col2_sim = st.columns(2)
        
        with col1_sim:
            valor_face_sim = st.number_input(
                "Valor de Face (R$):", 
                min_value=0.01, 
                value=1000.0, 
                step=10.0,
                key="vf_sim"
            )
            
            cupom_anual_sim = st.number_input(
                "Cupom Anual (R$):", 
                min_value=0.0, 
                value=80.0, 
                step=5.0,
                key="cupom_sim"
            )
            
            maturidade_sim = st.number_input(
                "Prazo até Vencimento (anos):", 
                min_value=0.1, 
                value=5.0, 
                step=0.5,
                key="mat_sim"
            )
        
        with col2_sim:
            ytm_sim = st.number_input(
                "YTM (% a.a.):", 
                min_value=0.0, 
                value=9.0, 
                step=0.1,
                key="ytm_sim"
            ) / 100
            
            freq_cupom_sim = st.selectbox(
                "Frequência de Pagamento:",
                [1, 2],
                format_func=lambda x: "Anual" if x == 1 else "Semestral",
                key="freq_sim"
            )
            
            variacao_taxa = st.slider(
                "Variação na taxa de juros (pontos percentuais):",
                min_value=-5.0,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="var_taxa_sim"
            ) / 100
        
        if st.button("Simular Variação de Preço", key="simular_variacao"):
            # Cálculos base
            preco_base = calcular_preco_titulo(valor_face_sim, cupom_anual_sim, ytm_sim, maturidade_sim, freq_cupom_sim)
            duration_mac_sim = calcular_duration_macaulay(valor_face_sim, cupom_anual_sim, ytm_sim, maturidade_sim, freq_cupom_sim)
            duration_mod_sim = calcular_duration_modificada(duration_mac_sim, ytm_sim, freq_cupom_sim)
            convexidade_sim = calcular_convexidade(valor_face_sim, cupom_anual_sim, ytm_sim, maturidade_sim, freq_cupom_sim)
            
            # Estimativa usando duration e convexidade
            var_estimada = estimar_variacao_preco(duration_mod_sim, convexidade_sim, variacao_taxa)
            preco_estimado = preco_base * (1 + var_estimada)
            
            # Estimativa usando apenas duration (sem convexidade)
            var_duration_only = -duration_mod_sim * variacao_taxa
            preco_duration_only = preco_base * (1 + var_duration_only)
            
            # Preço real para comparação
            ytm_novo = ytm_sim + variacao_taxa
            if ytm_novo > 0:
                preco_real = calcular_preco_titulo(valor_face_sim, cupom_anual_sim, ytm_novo, maturidade_sim, freq_cupom_sim)
                
                # Cálculo dos erros
                erro_duration_only = abs(preco_duration_only - preco_real)
                erro_duration_conv = abs(preco_estimado - preco_real)
                erro_pct_duration = (erro_duration_only / preco_real) * 100
                erro_pct_conv = (erro_duration_conv / preco_real) * 100
                
                # Mostrar resultados
                st.subheader("Resultados da Simulação")
                
                col1_res, col2_res, col3_res, col4_res = st.columns(4)
                
                with col1_res:
                    st.metric(
                        "Preço Original",
                        f"R$ {preco_base:.2f}"
                    )
                
                with col2_res:
                    diferenca_real = preco_real - preco_base
                    st.metric(
                        "Preço Real",
                        f"R$ {preco_real:.2f}",
                        f"{diferenca_real:.2f}"
                    )
                
                with col3_res:
                    diferenca_duration = preco_duration_only - preco_base
                    st.metric(
                        "Estimativa (Duration)",
                        f"R$ {preco_duration_only:.2f}",
                        f"{diferenca_duration:.2f}"
                    )
                    st.caption(f"Erro vs real: {erro_pct_duration:.3f}%")
                
                with col4_res:
                    diferenca_conv = preco_estimado - preco_base
                    st.metric(
                        "Estimativa (Dur+Conv)",
                        f"R$ {preco_estimado:.2f}",
                        f"{diferenca_conv:.2f}"
                    )
                    st.caption(f"Erro vs real: {erro_pct_conv:.3f}%")
                
                # Tabela comparativa detalhada
                st.subheader("Análise Comparativa Detalhada")
                
                cenarios_variacao = [-0.03, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.03]
                dados_comparacao = []
                
                for var in cenarios_variacao:
                    ytm_cenario = ytm_sim + var
                    if ytm_cenario > 0:
                        # Preço real
                        preco_real_cenario = calcular_preco_titulo(valor_face_sim, cupom_anual_sim, ytm_cenario, maturidade_sim, freq_cupom_sim)
                        
                        # Estimativa só com duration
                        var_dur_only = -duration_mod_sim * var
                        preco_dur_only = preco_base * (1 + var_dur_only)
                        erro_dur = abs(preco_dur_only - preco_real_cenario)
                        erro_pct_dur = (erro_dur / preco_real_cenario) * 100
                        
                        # Estimativa com duration + convexidade
                        var_dur_conv = estimar_variacao_preco(duration_mod_sim, convexidade_sim, var)
                        preco_dur_conv = preco_base * (1 + var_dur_conv)
                        erro_conv = abs(preco_dur_conv - preco_real_cenario)
                        erro_pct_conv = (erro_conv / preco_real_cenario) * 100
                        
                        dados_comparacao.append({
                            'Δ YTM': f"{var*100:+.1f}%",
                            'YTM Final': f"{ytm_cenario*100:.2f}%",
                            'Preço Real': f"R$ {preco_real_cenario:.2f}",
                            'Duration (R$)': f"R$ {preco_dur_only:.2f}",
                            'Erro Duration': f"{erro_pct_dur:.3f}%",
                            'Dur+Conv (R$)': f"R$ {preco_dur_conv:.2f}",
                            'Erro Dur+Conv': f"{erro_pct_conv:.3f}%"
                        })
                
                if dados_comparacao:
                    df_comp = pd.DataFrame(dados_comparacao)
                    st.dataframe(df_comp, use_container_width=True)
                    
                    # Gráfico comparativo
                    st.subheader("Visualização dos Erros de Estimativa")
                    
                    variacoes = [float(d['Δ YTM'].replace('%', '').replace('+', '')) for d in dados_comparacao]
                    erros_duration = [float(d['Erro Duration'].replace('%', '')) for d in dados_comparacao]
                    erros_conv = [float(d['Erro Dur+Conv'].replace('%', '')) for d in dados_comparacao]
                    
                    fig_erros = go.Figure()
                    
                    fig_erros.add_trace(go.Scatter(
                        x=variacoes,
                        y=erros_duration,
                        mode='lines+markers',
                        name='Erro Duration Apenas',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_erros.add_trace(go.Scatter(
                        x=variacoes,
                        y=erros_conv,
                        mode='lines+markers',
                        name='Erro Duration + Convexidade',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_erros.update_layout(
                        title="Precisão das Estimativas vs Variação da Taxa",
                        xaxis_title="Variação YTM (%)",
                        yaxis_title="Erro Absoluto (%)",
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig_erros, use_container_width=True)
                    
                    st.write("**Conclusões:**")
                    st.write("• Duration sozinha é boa para pequenas variações")
                    st.write("• Convexidade melhora muito a precisão para variações maiores")
                    st.write("• Para variações > 2%, os erros aumentam significativamente")
                    st.write("• A convexidade é especialmente importante para títulos de longo prazo")
                    
            else:
                st.error("YTM resultante é negativa. Escolha uma variação menor.")
    
    with tab4:
        st.subheader("Análise de Sensibilidade")
        st.write("Analise como mudanças nas taxas de juros afetam o preço do título")
        
        # Parâmetros para análise
        col1, col2 = st.columns(2)
        
        with col1:
            valor_face_sens = st.number_input("Valor de Face (R$):", value=1000.0, key="vf_sens")
            cupom_anual_sens = st.number_input("Cupom Anual (R$):", value=80.0, key="cupom_sens")
            maturidade_sens = st.number_input("Prazo (anos):", value=5.0, key="mat_sens")
        
        with col2:
            ytm_base_sens = st.number_input("YTM Base (%):", value=8.0, key="ytm_base") / 100
            range_taxa = st.slider("Range de análise (+/- pontos %):", 1, 10, 5, key="range_sens")
            freq_sens = st.selectbox("Frequência:", [1, 2], format_func=lambda x: "Anual" if x == 1 else "Semestral", key="freq_sens")
        
        if st.button("Gerar Análise de Sensibilidade", key="gerar_sens"):
            # Range de taxas
            taxa_min = max(0.001, ytm_base_sens - range_taxa/100)
            taxa_max = ytm_base_sens + range_taxa/100
            taxas = np.linspace(taxa_min, taxa_max, 100)
            
            # Calcular preços para diferentes taxas
            precos = []
            for taxa in taxas:
                preco = calcular_preco_titulo(valor_face_sens, cupom_anual_sens, taxa, maturidade_sens, freq_sens)
                precos.append(preco)
            
            # Gráfico de sensibilidade
            fig_sens = go.Figure()
            
            fig_sens.add_trace(go.Scatter(
                x=taxas * 100,
                y=precos,
                mode='lines',
                name='Preço do Título',
                line=dict(width=3, color='blue')
            ))
            
            # Marcar ponto atual
            preco_atual = calcular_preco_titulo(valor_face_sens, cupom_anual_sens, ytm_base_sens, maturidade_sens, freq_sens)
            fig_sens.add_trace(go.Scatter(
                x=[ytm_base_sens * 100],
                y=[preco_atual],
                mode='markers',
                name=f'YTM Atual ({ytm_base_sens*100:.1f}%)',
                marker=dict(size=12, color='red')
            ))
            
            # Linha do valor de face
            fig_sens.add_hline(y=valor_face_sens, line_dash="dash", line_color="green", 
                              annotation_text=f"Valor de Face: R$ {valor_face_sens}")
            
            fig_sens.update_layout(
                title="Análise de Sensibilidade: Preço vs Taxa de Juros",
                xaxis_title="YTM (%)",
                yaxis_title="Preço (R$)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_sens, use_container_width=True)
            
            # Comparação de diferentes maturidades
            st.subheader("Sensibilidade por Maturidade")
            
            maturidades = [1, 3, 5, 10, 20]
            fig_mat = go.Figure()
            
            for mat in maturidades:
                precos_mat = []
                for taxa in taxas:
                    preco_mat = calcular_preco_titulo(valor_face_sens, cupom_anual_sens, taxa, mat, freq_sens)
                    precos_mat.append(preco_mat)
                
                fig_mat.add_trace(go.Scatter(
                    x=taxas * 100,
                    y=precos_mat,
                    mode='lines',
                    name=f'{mat} anos'
                ))
            
            fig_mat.update_layout(
                title="Sensibilidade por Prazo até Vencimento",
                xaxis_title="YTM (%)",
                yaxis_title="Preço (R$)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_mat, use_container_width=True)
            
            # Tabela de cenários
            st.subheader("Cenários de Taxa de Juros")
            
            cenarios_taxa = [ytm_base_sens - 0.02, ytm_base_sens - 0.01, ytm_base_sens, ytm_base_sens + 0.01, ytm_base_sens + 0.02]
            cenarios_data = []
            
            for taxa_cenario in cenarios_taxa:
                if taxa_cenario > 0:
                    preco_cenario = calcular_preco_titulo(valor_face_sens, cupom_anual_sens, taxa_cenario, maturidade_sens, freq_sens)
                    variacao_preco = ((preco_cenario - preco_atual) / preco_atual) * 100
                    
                    cenarios_data.append({
                        'Cenário': f"YTM {taxa_cenario*100:.1f}%",
                        'Preço (R$)': f"{preco_cenario:.2f}",
                        'Variação (%)': f"{variacao_preco:+.2f}%",
                        'Ganho/Perda (R$)': f"R$ {preco_cenario - preco_atual:+.2f}"
                    })
            
            df_cenarios = pd.DataFrame(cenarios_data)
            st.table(df_cenarios)
    
    with tab5:
        st.subheader("Comparação de Títulos")
        st.write("Compare diferentes títulos do Tesouro Direto")
        
        # Definir títulos para comparação
        titulos_exemplos = {
            "Tesouro Prefixado 2027": {
                "valor_face": 1000,
                "cupom": 0,
                "maturidade": 3,
                "freq": 1,
                "ytm": 10.5
            },
            "Tesouro IPCA+ 2029": {
                "valor_face": 1000,
                "cupom": 0,
                "maturidade": 5,
                "freq": 1,
                "ytm": 5.8
            },
            "Tesouro IPCA+ Juros 2035": {
                "valor_face": 1000,
                "cupom": 60,
                "maturidade": 11,
                "freq": 2,
                "ytm": 5.8
            },
            "Tesouro Selic 2027": {
                "valor_face": 1000,
                "cupom": 0,
                "maturidade": 3,
                "freq": 1,
                "ytm": 11.75
            }
        }
        
        st.write("**Títulos Selecionados para Comparação:**")
        
        resultados_comparacao = []
        
        for nome, params in titulos_exemplos.items():
            preco = calcular_preco_titulo(
                params["valor_face"], params["cupom"], params["ytm"]/100, 
                params["maturidade"], params["freq"]
            )
            
            duration_mac = calcular_duration_macaulay(
                params["valor_face"], params["cupom"], params["ytm"]/100, 
                params["maturidade"], params["freq"]
            )
            
            duration_mod = calcular_duration_modificada(duration_mac, params["ytm"]/100, params["freq"])
            
            convexidade = calcular_convexidade(
                params["valor_face"], params["cupom"], params["ytm"]/100, 
                params["maturidade"], params["freq"]
            )
            
            current_yield = calcular_current_yield(params["cupom"], preco) if params["cupom"] > 0 else 0
            
            resultados_comparacao.append({
                "Título": nome,
                "Preço (R$)": f"{preco:.2f}",
                "YTM (%)": f"{params['ytm']:.2f}%",
                "Current Yield (%)": f"{current_yield*100:.2f}%" if current_yield > 0 else "N/A",
                "Duration Mod.": f"{duration_mod:.2f}",
                "Convexidade": f"{convexidade:.4f}",
                "Prazo (anos)": params["maturidade"]
            })
        
        df_comparacao = pd.DataFrame(resultados_comparacao)
        st.dataframe(df_comparacao, use_container_width=True)
        
        # Gráfico comparativo de duration
        st.subheader("Comparação de Risco (Duration Modificada)")
        
        nomes = [r["Título"] for r in resultados_comparacao]
        durations = [float(r["Duration Mod."]) for r in resultados_comparacao]
        
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Bar(
            x=nomes,
            y=durations,
            text=[f'{d:.2f}' for d in durations],
            textposition='auto',
            marker_color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        ))
        
        fig_comp.update_layout(
            title="Risco de Taxa de Juros por Título (Duration Modificada)",
            xaxis_title="Título",
            yaxis_title="Duration Modificada",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Simulação de cenário
        st.subheader("Simulação de Cenário de Taxa")
        
        variacao_simulacao = st.slider(
            "Variação nas taxas de juros (pontos percentuais):",
            min_value=-3.0,
            max_value=3.0,
            value=1.0,
            step=0.1
        ) / 100
        
        if st.button("Simular Cenário", key="simular_cenario"):
            st.write(f"**Cenário: Variação de {variacao_simulacao*100:+.1f} pontos percentuais nas taxas**")
            
            simulacao_data = []
            
            for nome, params in titulos_exemplos.items():
                # Preço atual
                preco_atual = calcular_preco_titulo(
                    params["valor_face"], params["cupom"], params["ytm"]/100, 
                    params["maturidade"], params["freq"]
                )
                
                # Preço no novo cenário
                nova_ytm = params["ytm"]/100 + variacao_simulacao
                if nova_ytm > 0:
                    preco_novo = calcular_preco_titulo(
                        params["valor_face"], params["cupom"], nova_ytm, 
                        params["maturidade"], params["freq"]
                    )
                    variacao_preco = ((preco_novo - preco_atual) / preco_atual) * 100
                    ganho_perda = preco_novo - preco_atual
                else:
                    preco_novo = 0
                    variacao_preco = 0
                    ganho_perda = 0
                
                simulacao_data.append({
                    "Título": nome,
                    "Preço Atual (R$)": f"{preco_atual:.2f}",
                    "Preço Novo (R$)": f"{preco_novo:.2f}",
                    "Variação (%)": f"{variacao_preco:+.2f}%",
                    "Ganho/Perda (R$)": f"R$ {ganho_perda:+.2f}"
                })
            
            df_simulacao = pd.DataFrame(simulacao_data)
            st.dataframe(df_simulacao, use_container_width=True)
            
            # Gráfico de ganhos e perdas
            ganhos_perdas = [float(r["Ganho/Perda (R$)"].replace("R$ ", "").replace("+", "")) for r in simulacao_data]
            cores = ['green' if gp >= 0 else 'red' for gp in ganhos_perdas]
            
            fig_gp = go.Figure()
            
            fig_gp.add_trace(go.Bar(
                x=nomes,
                y=ganhos_perdas,
                marker_color=cores,
                text=[f'R$ {gp:+.2f}' for gp in ganhos_perdas],
                textposition='auto'
            ))
            
            fig_gp.add_hline(y=0, line_dash="dash", line_color="white")
            
            fig_gp.update_layout(
                title=f"Impacto da Variação de {variacao_simulacao*100:+.1f}% nas Taxas",
                xaxis_title="Título",
                yaxis_title="Ganho/Perda (R$)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_gp, use_container_width=True)
    
    # Seção educativa
    with st.expander("Conceitos Importantes sobre Títulos de Renda Fixa"):
        st.markdown("""
        ### Conceitos Importantes:
        
        **Yield to Maturity (YTM):**
        - Taxa de retorno se o título for mantido até o vencimento
        - Considera todos os fluxos de caixa futuros
        - Taxa que iguala o preço de mercado ao valor presente dos fluxos
        
        **Duration de Macaulay:**
        - Prazo médio ponderado dos fluxos de caixa
        - Medida em anos
        - Indica o "prazo efetivo" do investimento
        
        **Duration Modificada:**
        - Medida de sensibilidade do preço às mudanças nas taxas
        - Duration Modificada = Duration Macaulay / (1 + YTM)
        - Variação percentual aproximada no preço para cada 1% de mudança na taxa
        
        **Convexidade:**
        - Medida de segunda ordem da relação preço-taxa
        - Melhora a estimativa de variação de preço
        - Convexidade positiva é desejável (proteção contra alta de juros)
        
        **Current Yield:**
        - Rendimento atual = Cupom Anual / Preço de Mercado
        - Ignora ganhos/perdas de capital
        - Útil para comparações rápidas
        
        ### Tipos de Títulos:
        
        **Par Value Bond:** Taxa de cupom = YTM (preço = valor de face)
        **Discount Bond:** Taxa de cupom < YTM (preço < valor de face)  
        **Premium Bond:** Taxa de cupom > YTM (preço > valor de face)
        
        ### Títulos do Tesouro Direto:
        
        **Tesouro Prefixado (LTN):** Taxa fixa, sem cupons intermediários
        **Tesouro IPCA+ Principal:** Protegido da inflação, sem cupons
        **Tesouro IPCA+ com Juros:** Protegido da inflação, com cupons semestrais
        **Tesouro Selic (LFT):** Indexado à taxa Selic, baixo risco de mercado
        """)

elif selected_tool == "Estrutura de Capital":
    st.subheader('Análise de Estrutura de Capital')
    
    st.write("""
    Esta ferramenta calcula o **WACC** (Custo Médio Ponderado de Capital), **CAPM** (Modelo de Precificação de Ativos Financeiros), 
    e analisa a estrutura de capital ótima considerando benefícios fiscais e custos de falência.
    """)
    
    # Abas para diferentes análises
    tab1, tab2, tab3 = st.tabs(["Calculadora WACC", "Análise CAPM", "Estrutura Ótima"])
    
    with tab1:
        st.subheader("Calculadora WACC")
        st.write("**WACC = (E/V × Re) + (D/V × Rd × (1-T))**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dados da Empresa:**")
            patrimonio_liquido = st.number_input("Patrimônio Líquido (E) - R$ milhões:", min_value=0.01, value=1000.0, step=10.0, key="pl_wacc")
            divida_total = st.number_input("Dívida Total (D) - R$ milhões:", min_value=0.0, value=500.0, step=10.0, key="div_wacc")
            
            st.write("**Custos de Capital:**")
            custo_equity = st.number_input("Custo do Capital Próprio (Re) - %:", min_value=0.0, value=12.0, step=0.1, key="re_wacc") / 100
            custo_divida = st.number_input("Custo da Dívida (Rd) - %:", min_value=0.0, value=8.0, step=0.1, key="rd_wacc") / 100
            
        with col2:
            st.write("**Impostos:**")
            taxa_imposto = st.number_input("Taxa de Imposto (T) - %:", min_value=0.0, max_value=100.0, value=34.0, step=1.0, key="tax_wacc") / 100
            
            # Cálculos automáticos
            valor_total = patrimonio_liquido + divida_total
            peso_equity = patrimonio_liquido / valor_total
            peso_divida = divida_total / valor_total
            custo_divida_liquido = custo_divida * (1 - taxa_imposto)
            
            st.write("**Pesos Calculados:**")
            st.write(f"• Peso do Patrimônio (E/V): {peso_equity:.1%}")
            st.write(f"• Peso da Dívida (D/V): {peso_divida:.1%}")
            st.write(f"• Custo da Dívida Líquido: {custo_divida_liquido:.2%}")
        
        if st.button("Calcular WACC", key="calc_wacc"):
            wacc = calcular_wacc(patrimonio_liquido, divida_total, valor_total, custo_equity, custo_divida, taxa_imposto)
            
            st.success(f"**WACC = {wacc:.2%}**")
            
            # Breakdown do cálculo
            st.subheader("Decomposição do WACC")
            
            componente_equity = peso_equity * custo_equity
            componente_divida = peso_divida * custo_divida_liquido
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Componente Equity", f"{componente_equity:.2%}", f"{peso_equity:.1%} × {custo_equity:.2%}")
            with col2:
                st.metric("Componente Dívida", f"{componente_divida:.2%}", f"{peso_divida:.1%} × {custo_divida_liquido:.2%}")
            with col3:
                st.metric("WACC Total", f"{wacc:.2%}", f"{componente_equity:.2%} + {componente_divida:.2%}")
            
            # Gráfico de pizza
            fig_wacc = go.Figure(data=[go.Pie(
                labels=['Capital Próprio', 'Capital de Terceiros'], 
                values=[patrimonio_liquido, divida_total],
                hole=0.3,
                textinfo='label+percent'
            )])
            fig_wacc.update_layout(
                title="Composição da Estrutura de Capital",
                template="plotly_dark"
            )
            st.plotly_chart(fig_wacc, use_container_width=True)
    
    with tab2:
        st.subheader("Análise CAPM")
        st.write("**Re = Rf + β × (Rm - Rf)**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parâmetros do CAPM:**")
            taxa_livre_risco = st.number_input("Taxa Livre de Risco (Rf) - %:", min_value=0.0, value=5.0, step=0.1, key="rf_capm") / 100
            beta = st.number_input("Beta (β):", min_value=0.0, value=1.2, step=0.1, key="beta_capm")
            retorno_mercado = st.number_input("Retorno do Mercado (Rm) - %:", min_value=0.0, value=12.0, step=0.1, key="rm_capm") / 100
            
        with col2:
            premio_risco = retorno_mercado - taxa_livre_risco
            custo_equity_capm = calcular_capm(taxa_livre_risco, beta, retorno_mercado)
            
            st.write("**Resultados:**")
            st.write(f"• Prêmio de Risco de Mercado: {premio_risco:.2%}")
            st.write(f"• Prêmio de Risco da Ação: {beta * premio_risco:.2%}")
            st.write(f"• **Custo do Capital Próprio: {custo_equity_capm:.2%}**")
        
        if st.button("Analisar Sensibilidade", key="sens_capm"):
            # Análise de sensibilidade do Beta
            betas = np.linspace(0.5, 2.0, 50)
            custos_equity = [calcular_capm(taxa_livre_risco, b, retorno_mercado) for b in betas]
            
            fig_beta = go.Figure()
            fig_beta.add_trace(go.Scatter(
                x=betas, 
                y=[c*100 for c in custos_equity], 
                mode='lines',
                name='Custo do Capital Próprio'
            ))
            fig_beta.add_vline(x=beta, line_dash="dash", line_color="red", 
                              annotation_text=f"Beta Atual: {beta}")
            fig_beta.update_layout(
                title="Sensibilidade do Custo de Capital ao Beta",
                xaxis_title="Beta",
                yaxis_title="Custo do Capital Próprio (%)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_beta, use_container_width=True)
            
            # Tabela de cenários
            st.subheader("Cenários de Beta")
            cenarios_beta = pd.DataFrame({
                'Cenário': ['Conservador', 'Atual', 'Agressivo'],
                'Beta': [0.8, beta, 1.5],
                'Custo Capital (%)': [f"{calcular_capm(taxa_livre_risco, 0.8, retorno_mercado):.2%}",
                                     f"{custo_equity:.2%}",
                                     f"{calcular_capm(taxa_livre_risco, 1.5, retorno_mercado):.2%}"]
            })
            st.table(cenarios_beta)
    
    with tab3:
        st.subheader("Análise da Estrutura de Capital Ótima")
        st.write("Esta análise considera o **trade-off** entre benefícios fiscais da dívida e custos de dificuldades financeiras.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parâmetros da Empresa:**")
            pl_otimo = st.number_input("Patrimônio Líquido Base - R$ milhões:", min_value=0.01, value=1000.0, step=10.0, key="pl_otimo")
            re_otimo = st.number_input("Custo do Capital Próprio - %:", min_value=0.0, value=12.0, step=0.1, key="re_otimo") / 100
            rd_otimo = st.number_input("Custo da Dívida - %:", min_value=0.0, value=8.0, step=0.1, key="rd_otimo") / 100
            
        with col2:
            st.write("**Parâmetros Fiscais e de Risco:**")
            tax_otimo = st.number_input("Taxa de Imposto - %:", min_value=0.0, max_value=100.0, value=34.0, step=1.0, key="tax_otimo") / 100
            custo_falencia = st.number_input("Taxa de Custo de Falência - %:", min_value=0.0, max_value=10.0, value=2.0, step=0.1, key="cf_otimo") / 100
            max_divida = st.number_input("Dívida Máxima a Analisar - R$ milhões:", min_value=0.0, value=2000.0, step=50.0, key="max_div")
        
        if st.button("Analisar Estrutura Ótima", key="analise_otima"):
            # Gerar range de valores de dívida
            valores_divida = np.linspace(0, max_divida, 100)
            
            # Analisar estrutura ótima
            df_analise = analisar_estrutura_otima(
                valores_divida, pl_otimo, re_otimo, rd_otimo, tax_otimo, custo_falencia
            )
            
            # Encontrar estrutura ótima (menor WACC)
            idx_otimo = df_analise['WACC'].idxmin()
            estrutura_otima = df_analise.iloc[idx_otimo]
            
            # Encontrar valor máximo da empresa
            idx_max_valor = df_analise['Valor_Empresa'].idxmax()
            max_valor_empresa = df_analise.iloc[idx_max_valor]
            
            # Mostrar resultados
            st.subheader("Estrutura de Capital Ótima")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "WACC Mínimo", 
                    f"{estrutura_otima['WACC']:.2%}",
                    f"Dívida: R$ {estrutura_otima['Divida']:.0f}M"
                )
            with col2:
                st.metric(
                    "Debt Ratio Ótimo", 
                    f"{estrutura_otima['Debt_Ratio']:.1%}",
                    f"Equity: {estrutura_otima['Equity_Ratio']:.1%}"
                )
            with col3:
                st.metric(
                    "Valor Máximo da Empresa", 
                    f"R$ {max_valor_empresa['Valor_Empresa']:.0f}M",
                    f"Tax Shield: R$ {max_valor_empresa['Tax_Shield']:.0f}M"
                )
            
            # Gráfico da análise de estrutura de capital
            fig_estrutura = go.Figure()
            
            # WACC
            fig_estrutura.add_trace(go.Scatter(
                x=df_analise['Debt_Ratio'] * 100,
                y=df_analise['WACC'] * 100,
                mode='lines',
                name='WACC (%)',
                line=dict(color='red', width=3),
                yaxis='y1'
            ))
            
            # Valor da Empresa (eixo secundário)
            fig_estrutura.add_trace(go.Scatter(
                x=df_analise['Debt_Ratio'] * 100,
                y=df_analise['Valor_Empresa'],
                mode='lines',
                name='Valor da Empresa (R$ M)',
                line=dict(color='green', width=3),
                yaxis='y2'
            ))
            
            # Marcar ponto ótimo
            fig_estrutura.add_trace(go.Scatter(
                x=[estrutura_otima['Debt_Ratio'] * 100],
                y=[estrutura_otima['WACC'] * 100],
                mode='markers',
                name='WACC Mínimo',
                marker=dict(color='red', size=12, symbol='star'),
                yaxis='y1'
            ))
            
            fig_estrutura.add_trace(go.Scatter(
                x=[max_valor_empresa['Debt_Ratio'] * 100],
                y=[max_valor_empresa['Valor_Empresa']],
                mode='markers',
                name='Valor Máximo',
                marker=dict(color='green', size=12, symbol='star'),
                yaxis='y2'
            ))
            
            fig_estrutura.update_layout(
                title="Análise da Estrutura de Capital Ótima",
                xaxis_title="Índice de Endividamento (%)",
                yaxis=dict(
                    title="WACC (%)",
                    side="left",
                    range=[df_analise['WACC'].min() * 80, df_analise['WACC'].max() * 120]
                ),
                yaxis2=dict(
                    title="Valor da Empresa (R$ Milhões)",
                    side="right",
                    overlaying="y"
                ),
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_estrutura, use_container_width=True)
            
            # Gráfico dos componentes (Tax Shield vs Custo de Falência)
            fig_componentes = go.Figure()
            
            fig_componentes.add_trace(go.Scatter(
                x=df_analise['Debt_Ratio'] * 100,
                y=df_analise['Tax_Shield'],
                mode='lines',
                name='Benefício Fiscal',
                line=dict(color='blue', width=2)
            ))
            
            fig_componentes.add_trace(go.Scatter(
                x=df_analise['Debt_Ratio'] * 100,
                y=df_analise['Custo_Falencia'],
                mode='lines',
                name='Custo de Falência',
                line=dict(color='orange', width=2)
            ))
            
            # Benefício líquido
            beneficio_liquido = df_analise['Tax_Shield'] - df_analise['Custo_Falencia']
            fig_componentes.add_trace(go.Scatter(
                x=df_analise['Debt_Ratio'] * 100,
                y=beneficio_liquido,
                mode='lines',
                name='Benefício Líquido',
                line=dict(color='purple', width=3, dash='dash')
            ))
            
            fig_componentes.update_layout(
                title="Trade-off: Benefícios Fiscais vs Custos de Falência",
                xaxis_title="Índice de Endividamento (%)",
                yaxis_title="Valor (R$ Milhões)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_componentes, use_container_width=True)
            
            # Tabela com cenários de estrutura de capital
            st.subheader("Cenários de Estrutura de Capital")
            
            # Selecionar alguns pontos chave
            cenarios_idx = [0, 25, 50, idx_otimo, 75, 99]
            df_cenarios = df_analise.iloc[cenarios_idx].copy()
            
            df_cenarios_display = pd.DataFrame({
                'Cenário': ['Sem Dívida', 'Baixo Endiv.', 'Médio Endiv.', 'Ótimo', 'Alto Endiv.', 'Máximo Endiv.'],
                'Dívida (R$ M)': df_cenarios['Divida'].round(0),
                'Debt Ratio': (df_cenarios['Debt_Ratio'] * 100).round(1).astype(str) + '%',
                'WACC': (df_cenarios['WACC'] * 100).round(2).astype(str) + '%',
                'Tax Shield (R$ M)': df_cenarios['Tax_Shield'].round(0),
                'Custo Falência (R$ M)': df_cenarios['Custo_Falencia'].round(0),
                'Valor Empresa (R$ M)': df_cenarios['Valor_Empresa'].round(0)
            })
            
            st.dataframe(df_cenarios_display, use_container_width=True)
            
            # Insights e recomendações
            st.subheader("Insights e Recomendações")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Análise dos Resultados:**")
                st.write(f"• **Estrutura ótima:** {estrutura_otima['Debt_Ratio']:.1%} de endividamento")
                st.write(f"• **WACC mínimo:** {estrutura_otima['WACC']:.2%}")
                st.write(f"• **Economia fiscal anual:** R$ {estrutura_otima['Tax_Shield']:.0f} milhões")
                st.write(f"• **Aumento de valor:** R$ {max_valor_empresa['Valor_Empresa'] - pl_otimo:.0f} milhões")
                
            with col2:
                st.write("**Considerações Importantes:**")
                st.write("• O modelo simplifica custos de falência")
                st.write("• Não considera flexibilidade financeira")
                st.write("• Assume custos de dívida constantes")
                st.write("• Ignora custos de agência")
            
            # Comparação com estrutura atual (se aplicável)
            if st.checkbox("Comparar com estrutura atual", key="comp_atual"):
                st.subheader("Comparação com Estrutura Atual")
                
                divida_atual = st.number_input(
                    "Dívida atual da empresa - R$ milhões:", 
                    min_value=0.0, 
                    value=500.0, 
                    step=10.0, 
                    key="div_atual"
                )
                
                # Calcular métricas atuais
                valor_atual = pl_otimo + divida_atual
                debt_ratio_atual = divida_atual / valor_atual
                wacc_atual = calcular_wacc(pl_otimo, divida_atual, valor_atual, re_otimo, rd_otimo, tax_otimo)
                
                # Comparação
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Debt Ratio Atual vs Ótimo",
                        f"{debt_ratio_atual:.1%}",
                        f"{(debt_ratio_atual - estrutura_otima['Debt_Ratio']) * 100:.1f} p.p."
                    )
                
                with col2:
                    st.metric(
                        "WACC Atual vs Ótimo",
                        f"{wacc_atual:.2%}",
                        f"{(wacc_atual - estrutura_otima['WACC']) * 100:.2f} p.p."
                    )
                
                with col3:
                    diferenca_valor = (estrutura_otima['Valor_Empresa'] - valor_atual)
                    st.metric(
                        "Potencial de Criação de Valor",
                        f"R$ {diferenca_valor:.0f}M",
                        f"{(diferenca_valor/valor_atual)*100:.1f}%" if valor_atual > 0 else "N/A"
                    )
                
                if diferenca_valor > 0:
                    st.success(f"**Recomendação:** A empresa pode criar R$ {diferenca_valor:.0f} milhões em valor ajustando sua estrutura de capital para o nível ótimo.")
                elif diferenca_valor < -50:  # Tolerância de R$ 50M
                    st.warning("**Atenção:** A empresa pode estar super-endividada. Considere reduzir o endividamento.")
                else:
                    st.info("**Status:** A estrutura atual está próxima do ótimo teórico.")
    
    # Seção de glossário e explicações
    with st.expander("Glossário e Conceitos"):
        st.markdown("""
        **WACC (Weighted Average Cost of Capital):**
        - Custo médio ponderado de capital
        - Representa o custo de financiamento da empresa
        - Usado como taxa de desconto para avaliação de projetos
        
        **CAPM (Capital Asset Pricing Model):**
        - Modelo para calcular o custo do capital próprio
        - Re = Rf + β × (Rm - Rf)
        - Considera risco sistemático (beta) e prêmio de risco
        
        **Beta (β):**
        - Medida de risco sistemático
        - β > 1: mais volátil que o mercado
        - β < 1: menos volátil que o mercado
        
        **Tax Shield (Benefício Fiscal):**
        - Economia fiscal devido aos juros da dívida
        - Tax Shield = Taxa de Imposto × Dívida
        
        **Trade-off Theory:**
        - Teoria que equilibra benefícios fiscais vs custos de falência
        - Existe uma estrutura de capital ótima
        - Maximiza valor da empresa / minimiza WACC
        """)
    
    # Aviso sobre limitações
    st.warning("""
    **Importante:** Esta ferramenta fornece estimativas baseadas em modelos teóricos. 
    Para decisões importantes de estrutura de capital, considere:
    - Análise detalhada do setor e concorrentes
    - Condições específicas da empresa
    - Flexibilidade financeira e acesso ao mercado
    - Consultoria com especialistas em finanças corporativas
    """)

elif selected_tool == "Black-Scholes-Merton":
    st.subheader('Calculadora Black-Scholes-Merton')

    st.write("Insira os parâmetros abaixo:")
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
    st.subheader('Calculadora de Gregas de Opções')

    st.write("Insira os parâmetros abaixo:")
    S = st.number_input("Preço do ativo subjacente (S):", min_value=0.0, step=0.01, key="S_g")
    K = st.number_input("Preço de exercício (K):", min_value=0.0, step=0.01, key="K_g")
    T = st.number_input("Tempo até o vencimento (T) em anos:", min_value=0.0, step=0.01, key="T_g")
    r = st.number_input("Taxa livre de risco (r) em %:", min_value=0.0, step=0.01, key="r_g") / 100
    sigma = st.number_input("Volatilidade (σ) em %:", min_value=0.0, step=0.01, key="sigma_g") / 100

    if st.button("Calcular Gregas"):
        d1 = (np.log(S / K) + (r + (sigma**2) / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)

        st.write(f"**Delta:** {delta:.4f}")
        st.write(f"**Gamma:** {gamma:.4f}")
        st.write(f"**Theta:** {theta:.4f}")
        st.write(f"**Vega:** {vega:.4f}")
        st.write(f"**Rho:** {rho:.4f}")

elif selected_tool == "Payoff de Opções":
    st.subheader('Payoff de Opções')

    st.write("Insira os detalhes da sua posição:")
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
    elif tipo_opcao == "Venda de Put":
        payoff = premio - np.maximum(strike - precos, 0)

    # Definindo intervalo expandido para o eixo y
    y_min = payoff.min() * 2  
    y_max = payoff.max() * 2  

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=precos, y=payoff, mode='lines', name='Payoff'))
    fig.update_layout(
        title="Gráfico de Payoff da Opção",
        xaxis_title="Preço do Ativo Subjacente",
        yaxis_title="Payoff",
        yaxis=dict(range=[y_min, y_max]),
        template="plotly_dark"
    )
    st.plotly_chart(fig)

elif selected_tool == "Simulador de Monte Carlo":
    st.subheader('Simulador de Monte Carlo para Opções')
    
    st.write("**Monte Carlo** é um método numérico que simula milhares de possíveis trajetórias do preço do ativo para calcular o valor da opção.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Parâmetros da Opção:**")
        S_mc = st.number_input("Preço do ativo subjacente (S):", min_value=0.01, value=100.0, step=0.01, key="S_mc")
        K_mc = st.number_input("Preço de exercício (K):", min_value=0.01, value=100.0, step=0.01, key="K_mc")
        T_mc = st.number_input("Tempo até vencimento (T) em anos:", min_value=0.01, value=0.25, step=0.01, key="T_mc")
        r_mc = st.number_input("Taxa livre de risco (r) em %:", min_value=0.0, value=5.0, step=0.01, key="r_mc") / 100
        sigma_mc = st.number_input("Volatilidade (σ) em %:", min_value=0.01, value=20.0, step=0.01, key="sigma_mc") / 100
        tipo_mc = st.radio("Tipo de opção:", ["Call", "Put"], key="tipo_mc")
    
    with col2:
        st.write("**Parâmetros da Simulação:**")
        n_sims = st.slider("Número de simulações:", min_value=1000, max_value=50000, value=10000, step=1000)
        n_steps = st.slider("Número de passos temporais:", min_value=50, max_value=500, value=100, step=50)
        mostrar_trajetorias = st.checkbox("Mostrar algumas trajetórias", value=True)
        n_trajetorias_plot = st.slider("Trajetórias a mostrar:", min_value=5, max_value=100, value=20, step=5)
    
    if st.button("Executar Simulação Monte Carlo", key="run_mc"):
        with st.spinner("Executando simulação..."):
            # Monte Carlo
            mc_price, paths, final_prices, payoffs = monte_carlo_option_pricing(
                S_mc, K_mc, T_mc, r_mc, sigma_mc, n_sims, n_steps, tipo_mc
            )
            
            # Black-Scholes para comparação
            bs_price = black_scholes(S_mc, K_mc, T_mc, r_mc, sigma_mc, tipo_mc)
            
            # Resultados
            st.subheader("Resultados da Simulação")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Preço Monte Carlo", f"R$ {mc_price:.4f}")
            with col2:
                st.metric("Preço Black-Scholes", f"R$ {bs_price:.4f}")
            with col3:
                diferenca_pct = ((mc_price - bs_price) / bs_price) * 100
                st.metric("Diferença (%)", f"{diferenca_pct:.2f}%")
            
            # Gráficos
            if mostrar_trajetorias:
                st.subheader("Trajetórias de Preço Simuladas")
                
                fig_paths = go.Figure()
                
                # Tempo
                time_steps = np.linspace(0, T_mc, n_steps + 1)
                
                # Plotar algumas trajetórias
                for i in range(min(n_trajetorias_plot, n_sims)):
                    fig_paths.add_trace(go.Scatter(
                        x=time_steps, 
                        y=paths[i], 
                        mode='lines', 
                        name=f'Trajetória {i+1}',
                        line=dict(width=1),
                        opacity=0.6,
                        showlegend=False
                    ))
                
                # Linha do strike
                fig_paths.add_hline(y=K_mc, line_dash="dash", line_color="red", 
                                   annotation_text=f"Strike: R$ {K_mc}")
                
                fig_paths.update_layout(
                    title="Simulação de Trajetórias de Preço (Movimento Browniano Geométrico)",
                    xaxis_title="Tempo (anos)",
                    yaxis_title="Preço do Ativo",
                    template="plotly_dark",
                    height=500
                )
                st.plotly_chart(fig_paths, use_container_width=True)
            
            # Histograma dos preços finais
            st.subheader("Distribuição dos Preços Finais")
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=final_prices,
                nbinsx=50,
                name="Preços Finais",
                opacity=0.7
            ))
            
            fig_hist.add_vline(x=K_mc, line_dash="dash", line_color="red", 
                              annotation_text=f"Strike: R$ {K_mc}")
            fig_hist.add_vline(x=np.mean(final_prices), line_dash="dot", line_color="yellow", 
                              annotation_text=f"Média: R$ {np.mean(final_prices):.2f}")
            
            fig_hist.update_layout(
                title="Distribuição dos Preços Finais do Ativo",
                xaxis_title="Preço Final",
                yaxis_title="Frequência",
                template="plotly_dark"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Histograma dos payoffs
            st.subheader("Distribuição dos Payoffs")
            
            fig_payoff = go.Figure()
            fig_payoff.add_trace(go.Histogram(
                x=payoffs,
                nbinsx=50,
                name="Payoffs",
                opacity=0.7
            ))
            
            fig_payoff.add_vline(x=np.mean(payoffs), line_dash="dot", line_color="yellow", 
                                annotation_text=f"Payoff Médio: R$ {np.mean(payoffs):.4f}")
            
            fig_payoff.update_layout(
                title="Distribuição dos Payoffs da Opção",
                xaxis_title="Payoff",
                yaxis_title="Frequência",
                template="plotly_dark"
            )
            st.plotly_chart(fig_payoff, use_container_width=True)
            
            # Estatísticas adicionais
            st.subheader("Estatísticas da Simulação")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Preços Finais:**")
                st.write(f"• Média: R$ {np.mean(final_prices):.2f}")
                st.write(f"• Mediana: R$ {np.median(final_prices):.2f}")
                st.write(f"• Desvio Padrão: R$ {np.std(final_prices):.2f}")
                st.write(f"• Mínimo: R$ {np.min(final_prices):.2f}")
                st.write(f"• Máximo: R$ {np.max(final_prices):.2f}")
            
            with col2:
                st.write("**Payoffs:**")
                st.write(f"• Payoff Médio: R$ {np.mean(payoffs):.4f}")
                st.write(f"• Payoff Mediano: R$ {np.median(payoffs):.4f}")
                st.write(f"• Desvio Padrão: R$ {np.std(payoffs):.4f}")
                st.write(f"• % In-the-Money: {(np.sum(payoffs > 0) / n_sims * 100):.1f}%")
                if tipo_mc.lower() == 'call':
                    st.write(f"• % Acima do Strike: {(np.sum(final_prices > K_mc) / n_sims * 100):.1f}%")
                else:
                    st.write(f"• % Abaixo do Strike: {(np.sum(final_prices < K_mc) / n_sims * 100):.1f}%")
            
            st.info(f"""
            **Interpretação:** A simulação Monte Carlo gerou {n_sims:,} cenários possíveis 
            para o preço do ativo em {T_mc} anos. O preço da opção é a média dos payoffs 
            descontada para valor presente. A diferença com Black-Scholes de {diferenca_pct:.2f}% 
            é esperada devido à natureza probabilística da simulação.
            """)

elif selected_tool == "Dividendos vs JCP":
    st.subheader('Política de Dividendos e Juros sobre Capital Próprio (JCP)')
    
    st.write("""
    Esta ferramenta analisa as diferentes estratégias de distribuição de resultados no Brasil, 
    comparando **dividendos** vs **JCP** sob a ótica do **tax shield** e otimização tributária.
    """)
    
    # Tabs para diferentes análises
    tab1, tab2, tab3 = st.tabs(["Análise Comparativa", "Simulador de Cenários", "Calculadora Rápida"])
    
    with tab1:
        st.subheader("Dividendos vs Juros sobre Capital Próprio")
        
        with st.expander("Entenda as Diferenças"):
            st.markdown("""
            **Dividendos:**
            - Isentos de imposto de renda para pessoa física (Lei 9.249/95)
            - NÃO são dedutíveis para a empresa (sem tax shield)
            - Distribuídos após o lucro líquido (pós-impostos)
            - Sem limite legal específico (desde que respeite reservas)
            
            **Juros sobre Capital Próprio (JCP):**
            - Dedutíveis do IR (25%) e CSLL (9%) da empresa = 34% tax shield
            - Retenção de 15% de IR na fonte para pessoa física
            - Limitados pela TJLP × Patrimônio Líquido
            - Podem ser convertidos em dividendos posteriormente
            
            **Tax Shield JCP = Valor JCP × 34% (para empresas no Lucro Real)**
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dados da Empresa:**")
            
            patrimonio_liquido_jcp = st.number_input(
                "Patrimônio Líquido (R$ milhões):", 
                min_value=0.01, 
                value=500.0, 
                step=10.0, 
                key="pl_jcp"
            )
            
            lucro_liquido_jcp = st.number_input(
                "Lucro Líquido Anual (R$ milhões):", 
                min_value=0.0, 
                value=100.0, 
                step=5.0, 
                key="ll_jcp"
            )
            
            valor_distribuir = st.number_input(
                "Valor a Distribuir (R$ milhões):", 
                min_value=0.01, 
                value=50.0, 
                step=1.0, 
                key="vd_jcp"
            )
            
            tjlp_anual = st.number_input(
                "TJLP Anual (%):", 
                min_value=0.0, 
                value=6.0, 
                step=0.1, 
                key="tjlp"
            ) / 100
        
        with col2:
            st.write("**Parâmetros Tributários:**")
            
            regime_tributario = st.selectbox(
                "Regime Tributário:",
                [
                    ("Lucro Real - Grandes Empresas", 0.25, 0.09),
                    ("Lucro Real - Pequenas/Médias", 0.15, 0.09),
                    ("Personalizado", 0.25, 0.09)
                ],
                format_func=lambda x: x[0],
                key="regime"
            )
            
            if regime_tributario[0] == "Personalizado":
                aliquota_ir = st.number_input("Alíquota IR (%):", min_value=0.0, value=25.0, step=1.0, key="ir_custom") / 100
                aliquota_csll = st.number_input("Alíquota CSLL (%):", min_value=0.0, value=9.0, step=1.0, key="csll_custom") / 100
            else:
                aliquota_ir = regime_tributario[1]
                aliquota_csll = regime_tributario[2]
            
            ir_retido = st.number_input(
                "IR Retido JCP (%):", 
                min_value=0.0, 
                max_value=30.0, 
                value=15.0, 
                step=1.0, 
                key="ir_retido"
            ) / 100
            
            aliquota_total = aliquota_ir + aliquota_csll
            st.write(f"**Alíquota Total Empresa:** {aliquota_total:.1%}")
        
        if st.button("Analisar Estratégias", key="analisar_jcp"):
            analise = analisar_dividendos_vs_jcp(
                valor_distribuir, patrimonio_liquido_jcp, tjlp_anual, lucro_liquido_jcp,
                aliquota_ir, aliquota_csll, ir_retido
            )
            
            # Mostrar limite de JCP
            st.subheader("Limite Legal de JCP")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Limite JCP", f"R$ {analise['limite_jcp']:.2f}M", f"TJLP {tjlp_anual:.1%} × PL")
            with col2:
                st.metric("Valor a Distribuir", f"R$ {valor_distribuir:.2f}M")
            with col3:
                pode_usar_jcp = valor_distribuir <= analise['limite_jcp']
                status = "Dentro do limite" if pode_usar_jcp else "Excede limite"
                cor_status = "green" if pode_usar_jcp else "red"
                st.markdown(f'<p style="color: {cor_status}; font-weight: bold;">{status}</p>', unsafe_allow_html=True)
            
            if not pode_usar_jcp:
                st.warning(f"O valor excede o limite legal em R$ {valor_distribuir - analise['limite_jcp']:.2f}M. Será necessária estratégia mista.")
            
            # Comparação dos cenários
            st.subheader("Comparação das Estratégias")
            
            cenarios_data = []
            
            # Cenário 1: Apenas Dividendos
            cenarios_data.append({
                'Estratégia': 'Apenas Dividendos',
                'Dividendos (R$ M)': f"{analise['cenario_dividendos']['valor']:.2f}",
                'JCP (R$ M)': "0,00",
                'Tax Shield (R$ M)': f"{analise['cenario_dividendos']['tax_shield']:.2f}",
                'Custo p/ Empresa (R$ M)': f"{analise['cenario_dividendos']['custo_empresa']:.2f}",
                'Líquido p/ Acionista (R$ M)': f"{analise['cenario_dividendos']['liquido_acionista']:.2f}",
                'Eficiência Fiscal': "0%"
            })
            
            # Cenário 2: Apenas JCP (se possível)
            if analise['jcp_possivel'] > 0:
                eficiencia_jcp = (analise['cenario_jcp']['tax_shield'] / analise['cenario_jcp']['valor']) * 100
                cenarios_data.append({
                    'Estratégia': 'Apenas JCP',
                    'Dividendos (R$ M)': "0,00",
                    'JCP (R$ M)': f"{analise['cenario_jcp']['valor']:.2f}",
                    'Tax Shield (R$ M)': f"{analise['cenario_jcp']['tax_shield']:.2f}",
                    'Custo p/ Empresa (R$ M)': f"{analise['cenario_jcp']['custo_empresa']:.2f}",
                    'Líquido p/ Acionista (R$ M)': f"{analise['cenario_jcp']['liquido_acionista']:.2f}",
                    'Eficiência Fiscal': f"{eficiencia_jcp:.1f}%"
                })
            
            # Cenário 3: Estratégia Mista
            if analise['dividendo_complementar'] > 0:
                eficiencia_mista = (analise['cenario_misto']['tax_shield'] / valor_distribuir) * 100
                cenarios_data.append({
                    'Estratégia': 'Estratégia Mista',
                    'Dividendos (R$ M)': f"{analise['dividendo_complementar']:.2f}",
                    'JCP (R$ M)': f"{analise['jcp_possivel']:.2f}",
                    'Tax Shield (R$ M)': f"{analise['cenario_misto']['tax_shield']:.2f}",
                    'Custo p/ Empresa (R$ M)': f"{analise['cenario_misto']['custo_empresa']:.2f}",
                    'Líquido p/ Acionista (R$ M)': f"{analise['cenario_misto']['liquido_acionista']:.2f}",
                    'Eficiência Fiscal': f"{eficiencia_mista:.1f}%"
                })
            
            df_cenarios = pd.DataFrame(cenarios_data)
            st.dataframe(df_cenarios, use_container_width=True)
            
            # Gráfico comparativo
            st.subheader("Comparação Visual")
            
            estrategias = [c['Estratégia'] for c in cenarios_data]
            tax_shields = [float(c['Tax Shield (R$ M)'].replace(',', '.')) for c in cenarios_data]
            custos_empresa = [float(c['Custo p/ Empresa (R$ M)'].replace(',', '.')) for c in cenarios_data]
            
            fig_comp = go.Figure()
            
            # Tax Shield
            fig_comp.add_trace(go.Bar(
                name='Tax Shield (Economia)',
                x=estrategias,
                y=tax_shields,
                text=[f'R$ {ts:.2f}M' for ts in tax_shields],
                textposition='auto',
                marker_color='green',
                opacity=0.8
            ))
            
            # Custo para empresa
            fig_comp.add_trace(go.Bar(
                name='Custo Real p/ Empresa',
                x=estrategias,
                y=custos_empresa,
                text=[f'R$ {ce:.2f}M' for ce in custos_empresa],
                textposition='auto',
                marker_color='red',
                opacity=0.8
            ))
            
            fig_comp.update_layout(
                title="Tax Shield vs Custo Real para a Empresa",
                xaxis_title="Estratégia",
                yaxis_title="Valor (R$ Milhões)",
                template="plotly_dark",
                barmode='group'
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Recomendação
            if analise['jcp_possivel'] > 0:
                economia_maxima = analise['cenario_misto']['tax_shield']
                percentual_economia = (economia_maxima / valor_distribuir) * 100
                
                st.success(f"""
                **Estratégia Recomendada:** {'Estratégia Mista' if analise['dividendo_complementar'] > 0 else 'JCP Puro'}
                
                - **Economia fiscal:** R$ {economia_maxima:.2f} milhões ({percentual_economia:.1f}%)
                - **JCP ótimo:** R$ {analise['jcp_possivel']:.2f} milhões
                """)
                
                if analise['dividendo_complementar'] > 0:
                    st.write(f"- **Dividendo complementar:** R$ {analise['dividendo_complementar']:.2f} milhões")
            else:
                st.warning("Com a TJLP atual, não é possível usar JCP para este valor de distribuição.")
    
    with tab2:
        st.subheader("Simulador de Cenários")
        st.write("Analise como mudanças na TJLP e no patrimônio líquido afetam a estratégia ótima")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parâmetros Base:**")
            pl_base = st.number_input("PL Base (R$ milhões):", value=500.0, step=50.0, key="pl_sim")
            distribuicao_base = st.number_input("Distribuição Base (R$ milhões):", value=50.0, step=5.0, key="dist_sim")
            
        with col2:
            st.write("**Cenários TJLP:**")
            tjlp_min = st.number_input("TJLP Mínima (%):", value=4.0, step=0.5, key="tjlp_min") / 100
            tjlp_max = st.number_input("TJLP Máxima (%):", value=10.0, step=0.5, key="tjlp_max") / 100
        
        if st.button("Simular Cenários", key="simular_cenarios"):
            # Simulação de diferentes TJLPs
            tjlps = np.linspace(tjlp_min, tjlp_max, 20)
            resultados_sim = []
            
            for tjlp in tjlps:
                limite = calcular_limite_jcp(pl_base, tjlp)
                jcp_possivel = min(distribuicao_base, limite)
                tax_shield = calcular_tax_shield_jcp(jcp_possivel, 0.25, 0.09)
                eficiencia = (tax_shield / distribuicao_base) * 100 if distribuicao_base > 0 else 0
                
                resultados_sim.append({
                    'TJLP': tjlp * 100,
                    'Limite JCP': limite,
                    'JCP Possível': jcp_possivel,
                    'Tax Shield': tax_shield,
                    'Eficiência': eficiencia
                })
            
            df_sim = pd.DataFrame(resultados_sim)
            
            # Gráficos de sensibilidade
            fig_sens = go.Figure()
            
            # Tax Shield
            fig_sens.add_trace(go.Scatter(
                x=df_sim['TJLP'],
                y=df_sim['Tax Shield'],
                mode='lines+markers',
                name='Tax Shield (R$ M)',
                line=dict(color='green', width=3)
            ))
            
            fig_sens.update_layout(
                title="Sensibilidade do Tax Shield à TJLP",
                xaxis_title="TJLP (%)",
                yaxis_title="Tax Shield (R$ Milhões)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_sens, use_container_width=True)
            
            # Tabela de resultados
            st.subheader("Resultados da Simulação")
            
            df_display = df_sim.copy()
            df_display['TJLP'] = df_display['TJLP'].apply(lambda x: f"{x:.1f}%")
            df_display['Limite JCP'] = df_display['Limite JCP'].apply(lambda x: f"R$ {x:.1f}M")
            df_display['JCP Possível'] = df_display['JCP Possível'].apply(lambda x: f"R$ {x:.1f}M")
            df_display['Tax Shield'] = df_display['Tax Shield'].apply(lambda x: f"R$ {x:.2f}M")
            df_display['Eficiência'] = df_display['Eficiência'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(df_display, use_container_width=True)
    
    with tab3:
        st.subheader("Calculadora Rápida de Tax Shield")
        st.write("Cálculo instantâneo do benefício fiscal do JCP")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            jcp_rapido = st.number_input(
                "Valor JCP (R$ mil):", 
                min_value=0.0, 
                value=1000.0, 
                step=100.0, 
                key="jcp_rapido"
            )
        
        with col2:
            regime_rapido = st.selectbox(
                "Regime Tributário:",
                [
                    ("Lucro Real - Grandes Empresas", 0.34),
                    ("Lucro Real - Pequenas/Médias", 0.24),
                    ("Lucro Presumido", 0.24),
                    ("Personalizado", 0.34)
                ],
                format_func=lambda x: f"{x[0]} ({x[1]:.0%})",
                key="regime_rapido"
            )
            
            if regime_rapido[0] == "Personalizado":
                aliquota_personalizada = st.number_input(
                    "Alíquota Total (%):", 
                    value=34.0, 
                    step=1.0, 
                    key="aliq_pers"
                ) / 100
                aliquota_usar = aliquota_personalizada
            else:
                aliquota_usar = regime_rapido[1]
        
        with col3:
            tax_shield_rapido = jcp_rapido * aliquota_usar
            liquido_acionista_rapido = jcp_rapido * (1 - 0.15)  # 15% IR retido
            custo_real_empresa = jcp_rapido - tax_shield_rapido
            
            st.metric("Tax Shield", f"R$ {tax_shield_rapido:,.0f}", f"{aliquota_usar:.0%} do JCP")
            st.metric("Líquido p/ Acionista", f"R$ {liquido_acionista_rapido:,.0f}", "Após 15% IR")
            st.metric("Custo Real Empresa", f"R$ {custo_real_empresa:,.0f}", f"-{aliquota_usar:.0%}")
        
        # Comparador rápido
        st.markdown("---")
        st.subheader("Comparador Rápido: JCP vs Dividendo Equivalente")
        
        valor_comparacao = st.number_input(
            "Valor para Comparação (R$ mil):", 
            value=1000.0, 
            step=100.0, 
            key="comp_rapido"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Distribuição via JCP:**")
            tax_shield_comp = valor_comparacao * aliquota_usar
            liquido_acionista_jcp = valor_comparacao * 0.85
            custo_empresa_jcp = valor_comparacao - tax_shield_comp
            
            st.write(f"• Tax Shield: R$ {tax_shield_comp:,.0f}")
            st.write(f"• Líquido Acionista: R$ {liquido_acionista_jcp:,.0f}")
            st.write(f"• Custo Real Empresa: R$ {custo_empresa_jcp:,.0f}")
        
        with col2:
            st.write("**Distribuição via Dividendos:**")
            # Para ter mesmo valor líquido no acionista
            dividendo_equivalente = liquido_acionista_jcp  # Dividendos são isentos
            
            st.write(f"• Tax Shield: R$ 0")
            st.write(f"• Líquido Acionista: R$ {dividendo_equivalente:,.0f}")
            st.write(f"• Custo Real Empresa: R$ {dividendo_equivalente:,.0f}")
        
        diferenca_custo = dividendo_equivalente - custo_empresa_jcp
        st.success(f"**Economia com JCP:** R$ {diferenca_custo:,.0f} ({(diferenca_custo/dividendo_equivalente)*100:.1f}%)")
    
    # Seção educativa
    with st.expander("Conceitos e Legislação"):
        st.markdown("""
        ### Marco Legal
        
        **Lei 9.249/95** - Criou os Juros sobre Capital Próprio no Brasil como forma de:
        - Equiparar o tratamento tributário entre capital próprio e de terceiros
        - Permitir dedutibilidade fiscal para remuneração do capital próprio
        - Incentivar a capitalização das empresas
        
        ### Cálculo do Limite JCP
        
        **Limite Anual = TJLP × Patrimônio Líquido**
        
        Onde:
        - **TJLP**: Taxa de Juros de Longo Prazo (divulgada pelo CMN)
        - **Patrimônio Líquido**: Saldo médio anual ou do último dia do exercício
        
        ### Tax Shield (Benefício Fiscal)
        
        **Tax Shield = JCP × (IR + CSLL)**
        
        Para empresas no Lucro Real:
        - **Grandes empresas**: 25% (IR) + 9% (CSLL) = 34%
        - **Pequenas/médias**: 15% (IR) + 9% (CSLL) = 24%
        
        ### Tributação para o Acionista
        
        **Pessoa Física:**
        - **JCP**: Retenção de 15% na fonte (definitivo)
        - **Dividendos**: Isentos de IR
        
        **Pessoa Jurídica:**
        - **JCP**: Tributação conforme regime (pode ser dedutível)
        - **Dividendos**: Isentos para beneficiária brasileira
        
        ### Aspectos Práticos
        
        - JCP pode ser convertido em dividendo na assembleia
        - Base de cálculo: patrimônio líquido do exercício anterior
        - Dedutibilidade limitada ao maior entre:
          - 50% do lucro líquido do exercício
          - 50% dos lucros acumulados e reservas de lucros
        """)
    
    # Aviso legal
    st.warning("""
    **Importante:** Esta ferramenta fornece simulações baseadas na legislação tributária vigente. 
    Para decisões definitivas sobre política de dividendos, consulte:
    - Contador ou consultor tributário especializado
    - Departamento jurídico da empresa
    - Análise específica do estatuto social
    - TJLP atualizada no site do Banco Central
    
    **Disclaimer:** A legislação tributária pode sofrer alterações. Sempre verifique as normas atuais.
    """)

elif selected_tool == "Economia do 'e'":
    st.subheader('A Economia do Número "e" e Crescimento Exponencial')
    
    st.write("""
    Esta ferramenta explora a **interpretação econômica do número e**, comparações entre 
    **capitalização discreta vs contínua**, e aplicações do **crescimento exponencial** em finanças.
    """)
    
    # Tabs para diferentes análises
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "O Número e", 
        "Capitalização: Discreta vs Contínua", 
        "Conversão de Taxas", 
        "Crescimento e Desconto",
        "Aplicações Econômicas"
    ])
    
    with tab1:
        st.subheader("Descobrindo o Número e ≈ 2.71828")
        
        st.write("""
        **Definição Matemática:** e = lim(m→∞) (1 + 1/m)^m
        
        **Interpretação Econômica:** Valor final de $1 investido a 100% ao ano 
        com capitalização contínua.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Calculadora do Número e:**")
            
            m_max = st.slider(
                "Número de períodos (m):",
                min_value=1,
                max_value=1000000,
                value=10000,
                step=1000,
                key="m_aprox"
            )
            
            # Calcular aproximações
            valores_m = [1, 2, 4, 12, 52, 365, 1000, 10000, 100000, m_max]
            aproximacoes = []
            
            for m in valores_m:
                aprox = calcular_e_aproximacao(m)
                aproximacoes.append({
                    'Períodos (m)': f"{m:,}",
                    'Equação': f"(1 + 1/{m})^{m}",
                    'Valor': f"{aprox:.8f}",
                    'Diferença de e': f"{abs(aprox - np.e):.8f}"
                })
            
            df_aprox = pd.DataFrame(aproximacoes)
            st.dataframe(df_aprox, use_container_width=True)
            
            # Valor exato de e
            st.success(f"**Valor exato de e:** {np.e:.10f}")
            st.info(f"**Aproximação com m={m_max:,}:** {calcular_e_aproximacao(m_max):.10f}")
        
        with col2:
            st.write("**Interpretação Financeira:**")
            
            principal_e = st.number_input("Principal inicial ($):", value=1.0, step=0.1, key="principal_e")
            
            # Diferentes frequências de capitalização
            freq_capitalizacao = [1, 2, 4, 12, 52, 365, 8760]  # anual, semestral, trimestral, mensal, semanal, diária, horária
            freq_nomes = ["Anual", "Semestral", "Trimestral", "Mensal", "Semanal", "Diária", "Horária"]
            
            st.write("**Efeito da Frequência de Capitalização (Taxa: 100% a.a.):**")
            
            resultados_freq = []
            for freq, nome in zip(freq_capitalizacao, freq_nomes):
                valor = valor_futuro_composto(principal_e, 1.0, freq, 1.0)  # 100% ao ano, 1 ano
                resultados_freq.append({
                    'Frequência': nome,
                    'Períodos/Ano': freq,
                    'Valor Final': f"${valor:.6f}",
                    'Múltiplo do Principal': f"{valor/principal_e:.6f}"
                })
            
            # Capitalização contínua
            valor_continuo = valor_futuro_continuo(principal_e, 1.0, 1.0)
            resultados_freq.append({
                'Frequência': 'Contínua',
                'Períodos/Ano': '∞',
                'Valor Final': f"${valor_continuo:.6f}",
                'Múltiplo do Principal': f"{valor_continuo/principal_e:.6f}"
            })
            
            df_freq = pd.DataFrame(resultados_freq)
            st.dataframe(df_freq, use_container_width=True)
            
            st.write(f"**Observe:** Com capitalização contínua, $1 se torna ${np.e:.6f} = e")
        
        # Gráfico da convergência para e
        st.subheader("Convergência para o Número e")
        
        m_values = np.logspace(0, 6, 100)  # De 1 a 1,000,000
        e_approx = [(1 + 1/m)**m for m in m_values]
        
        fig_e = go.Figure()
        
        fig_e.add_trace(go.Scatter(
            x=m_values,
            y=e_approx,
            mode='lines',
            name='(1 + 1/m)^m',
            line=dict(color='blue', width=2)
        ))
        
        fig_e.add_hline(y=np.e, line_dash="dash", line_color="red", 
                       annotation_text=f"e = {np.e:.6f}")
        
        fig_e.update_layout(
            title="Convergência da Equação (1 + 1/m)^m para o Número e",
            xaxis_title="Número de Períodos (m)",
            yaxis_title="Valor da Expressão",
            xaxis_type="log",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_e, use_container_width=True)
    
    with tab2:
        st.subheader("Capitalização Discreta vs Contínua")
        
        st.write("""
        **Capitalização Discreta:** V = A(1 + i)^t  
        **Capitalização Contínua:** V = Ae^(rt)
        
        Onde i = taxa efetiva por período, r = taxa contínua, t = tempo
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parâmetros do Investimento:**")
            
            principal = st.number_input("Capital Inicial ($):", min_value=0.01, value=1000.0, step=100.0, key="principal_comp")
            taxa_anual = st.number_input("Taxa Anual (%):", min_value=0.0, value=10.0, step=0.5, key="taxa_comp") / 100
            tempo_anos = st.number_input("Tempo (anos):", min_value=0.1, value=5.0, step=0.5, key="tempo_comp")
            
        with col2:
            st.write("**Frequências de Capitalização:**")
            
            frequencias = {
                "Anual": 1,
                "Semestral": 2, 
                "Trimestral": 4,
                "Mensal": 12,
                "Semanal": 52,
                "Diária": 365,
                "Contínua": "∞"
            }
            
            resultados_comp = []
            
            for nome, freq in frequencias.items():
                if nome == "Contínua":
                    valor = valor_futuro_continuo(principal, taxa_anual, tempo_anos)
                    formula = f"${principal:,.0f} × e^({taxa_anual:.3f} × {tempo_anos})"
                else:
                    valor = valor_futuro_composto(principal, taxa_anual, freq, tempo_anos)
                    formula = f"${principal:,.0f} × (1 + {taxa_anual:.3f}/{freq})^({freq} × {tempo_anos})"
                
                resultados_comp.append({
                    'Capitalização': nome,
                    'Frequência': freq if nome != "Contínua" else "∞",
                    'Valor Final': f"${valor:,.2f}",
                    'Ganho': f"${valor - principal:,.2f}",
                    'Equação': formula
                })
            
            df_comp = pd.DataFrame(resultados_comp)
            st.dataframe(df_comp, use_container_width=True)
        
        # Comparação gráfica
        st.subheader("Comparação Gráfica: Crescimento ao Longo do Tempo")
        
        tempo_range = np.linspace(0, tempo_anos, 100)
        
        fig_comp = go.Figure()
        
        # Capitalização anual (discreta)
        valores_anual = [valor_futuro_discreto(principal, taxa_anual, t) for t in tempo_range]
        fig_comp.add_trace(go.Scatter(
            x=tempo_range,
            y=valores_anual,
            mode='lines',
            name='Capitalização Anual',
            line=dict(color='blue', width=2)
        ))
        
        # Capitalização mensal
        valores_mensal = [valor_futuro_composto(principal, taxa_anual, 12, t) for t in tempo_range]
        fig_comp.add_trace(go.Scatter(
            x=tempo_range,
            y=valores_mensal,
            mode='lines',
            name='Capitalização Mensal',
            line=dict(color='green', width=2)
        ))
        
        # Capitalização contínua
        valores_continuo = [valor_futuro_continuo(principal, taxa_anual, t) for t in tempo_range]
        fig_comp.add_trace(go.Scatter(
            x=tempo_range,
            y=valores_continuo,
            mode='lines',
            name='Capitalização Contínua',
            line=dict(color='red', width=3, dash='dash')
        ))
        
        fig_comp.update_layout(
            title="Crescimento do Capital: Diferentes Frequências de Capitalização",
            xaxis_title="Tempo (anos)",
            yaxis_title="Valor ($)",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Análise das diferenças
        st.subheader("Análise das Diferenças")
        
        valor_anual = valor_futuro_discreto(principal, taxa_anual, tempo_anos)
        valor_continuo_final = valor_futuro_continuo(principal, taxa_anual, tempo_anos)
        diferenca = valor_continuo_final - valor_anual
        percentual_diferenca = (diferenca / valor_anual) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Capitalização Anual", f"${valor_anual:,.2f}")
        
        with col2:
            st.metric("Capitalização Contínua", f"${valor_continuo_final:,.2f}")
        
        with col3:
            st.metric("Diferença", f"${diferenca:,.2f}", f"{percentual_diferenca:.2f}%")
        
        st.info(f"""
        **Conclusão:** Com uma taxa de {taxa_anual:.1%} ao ano por {tempo_anos} anos, 
        a capitalização contínua gera ${diferenca:,.2f} a mais ({percentual_diferenca:.2f}%) 
        que a capitalização anual simples.
        """)
    
    with tab3:
        st.subheader("Conversão entre Taxas Discretas e Contínuas")
        
        st.write("""
        **Equações de Conversão:**
        - **Discreta → Contínua:** r = ln(1 + i)
        - **Contínua → Discreta:** i = e^r - 1
        
        Onde i = taxa discreta e r = taxa contínua
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Converter Taxa Discreta para Contínua:**")
            
            taxa_discreta = st.number_input(
                "Taxa Discreta (% a.a.):",
                min_value=0.0,
                value=10.0,
                step=0.1,
                key="taxa_disc"
            ) / 100
            
            taxa_continua_conv = taxa_discreta_para_continua(taxa_discreta)
            
            st.success(f"""
            **Taxa Discreta:** {taxa_discreta:.4%}  
            **Taxa Contínua:** {taxa_continua_conv:.4%}  
            **Equação:** ln(1 + {taxa_discreta:.4f}) = {taxa_continua_conv:.6f}
            """)
            
        with col2:
            st.write("**Converter Taxa Contínua para Discreta:**")
            
            taxa_continua_input = st.number_input(
                "Taxa Contínua (% a.a.):",
                min_value=0.0,
                value=9.53,
                step=0.01,
                key="taxa_cont"
            ) / 100
            
            taxa_discreta_conv = taxa_continua_para_discreta(taxa_continua_input)
            
            st.success(f"""
            **Taxa Contínua:** {taxa_continua_input:.4%}  
            **Taxa Discreta:** {taxa_discreta_conv:.4%}  
            **Equação:** e^{taxa_continua_input:.6f} - 1 = {taxa_discreta_conv:.6f}
            """)
        
        # Tabela de equivalências
        st.subheader("Tabela de Equivalências")
        
        taxas_exemplo = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
        equivalencias = []
        
        for taxa_d in taxas_exemplo:
            taxa_c = taxa_discreta_para_continua(taxa_d)
            equivalencias.append({
                'Taxa Discreta': f"{taxa_d:.1%}",
                'Taxa Contínua': f"{taxa_c:.3%}",
                'Diferença': f"{(taxa_d - taxa_c)*100:.2f} p.p."
            })
        
        df_equiv = pd.DataFrame(equivalencias)
        st.dataframe(df_equiv, use_container_width=True)
        
        # Gráfico de comparação
        fig_conv = go.Figure()
        
        taxas_range = np.linspace(0, 0.30, 100)
        taxas_continuas = [taxa_discreta_para_continua(td) for td in taxas_range]
        
        fig_conv.add_trace(go.Scatter(
            x=taxas_range * 100,
            y=[tc * 100 for tc in taxas_continuas],
            mode='lines',
            name='Taxa Contínua',
            line=dict(color='blue', width=3)
        ))
        
        # Linha de 45 graus (onde seriam iguais)
        fig_conv.add_trace(go.Scatter(
            x=taxas_range * 100,
            y=taxas_range * 100,
            mode='lines',
            name='Linha de Igualdade',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_conv.update_layout(
            title="Relação entre Taxas Discretas e Contínuas",
            xaxis_title="Taxa Discreta (%)",
            yaxis_title="Taxa Contínua (%)",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_conv, use_container_width=True)
        
        st.info("**Observação:** A taxa contínua é sempre menor que a taxa discreta equivalente.")
    
    with tab4:
        st.subheader("Crescimento Exponencial e Desconto")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Valor Futuro (Crescimento):**")
            
            principal_crescimento = st.number_input("Valor Presente ($):", value=1000.0, step=100.0, key="principal_cresc")
            taxa_crescimento = st.number_input("Taxa de Crescimento (% a.a.):", value=8.0, step=0.5, key="taxa_cresc") / 100
            tempo_crescimento = st.number_input("Tempo (anos):", value=10.0, step=1.0, key="tempo_cresc")
            
            if st.button("Calcular Valor Futuro", key="calc_vf"):
                # Crescimento discreto
                vf_discreto = valor_futuro_discreto(principal_crescimento, taxa_crescimento, tempo_crescimento)
                
                # Crescimento contínuo
                vf_continuo = valor_futuro_continuo(principal_crescimento, taxa_crescimento, tempo_crescimento)
                
                st.write("**Resultados:**")
                st.write(f"• **Discreto:** ${vf_discreto:,.2f}")
                st.write(f"• **Contínuo:** ${vf_continuo:,.2f}")
                st.write(f"• **Diferença:** ${vf_continuo - vf_discreto:,.2f}")
                
        with col2:
            st.write("**Valor Presente (Desconto):**")
            
            valor_futuro_desc = st.number_input("Valor Futuro ($):", value=2000.0, step=100.0, key="vf_desc")
            taxa_desconto = st.number_input("Taxa de Desconto (% a.a.):", value=8.0, step=0.5, key="taxa_desc") / 100
            tempo_desconto = st.number_input("Tempo até recebimento (anos):", value=10.0, step=1.0, key="tempo_desc")
            
            if st.button("Calcular Valor Presente", key="calc_vp"):
                # Desconto discreto
                vp_discreto = valor_presente_discreto(valor_futuro_desc, taxa_desconto, tempo_desconto)
                
                # Desconto contínuo
                vp_continuo = valor_presente_continuo(valor_futuro_desc, taxa_desconto, tempo_desconto)
                
                st.write("**Resultados:**")
                st.write(f"• **Discreto:** ${vp_discreto:,.2f}")
                st.write(f"• **Contínuo:** ${vp_continuo:,.2f}")
                st.write(f"• **Diferença:** ${abs(vp_continuo - vp_discreto):,.2f}")
        
        # Comparação gráfica de crescimento vs linear
        st.subheader("Crescimento Exponencial vs Linear")
        
        col1, col2 = st.columns(2)
        
        with col1:
            principal_comp = st.number_input("Capital Inicial ($):", value=1000.0, key="cap_comp")
            taxa_exp = st.number_input("Taxa Exponencial (% a.a.):", value=7.0, key="taxa_exp") / 100
            taxa_linear = st.number_input("Taxa Linear ($ por ano):", value=70.0, key="taxa_lin")
            anos_simulacao = st.slider("Anos de Simulação:", 1, 30, 20, key="anos_sim")
        
        with col2:
            tempo_sim = np.linspace(0, anos_simulacao, anos_simulacao * 4)
            
            # Crescimento exponencial
            valores_exp = [valor_futuro_continuo(principal_comp, taxa_exp, t) for t in tempo_sim]
            
            # Crescimento linear
            valores_linear = [principal_comp + taxa_linear * t for t in tempo_sim]
            
            fig_cresc = go.Figure()
            
            fig_cresc.add_trace(go.Scatter(
                x=tempo_sim,
                y=valores_exp,
                mode='lines',
                name=f'Exponencial ({taxa_exp:.1%})',
                line=dict(color='red', width=3)
            ))
            
            fig_cresc.add_trace(go.Scatter(
                x=tempo_sim,
                y=valores_linear,
                mode='lines',
                name=f'Linear (${taxa_linear}/ano)',
                line=dict(color='blue', width=3)
            ))
            
            fig_cresc.update_layout(
                title="Crescimento Exponencial vs Linear",
                xaxis_title="Tempo (anos)",
                yaxis_title="Valor ($)",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig_cresc, use_container_width=True)
        
        # Ponto de intersecção
        valor_exp_final = valor_futuro_continuo(principal_comp, taxa_exp, anos_simulacao)
        valor_linear_final = principal_comp + taxa_linear * anos_simulacao
        
        st.info(f"""
        **Após {anos_simulacao} anos:**
        - **Crescimento Exponencial:** ${valor_exp_final:,.2f}
        - **Crescimento Linear:** ${valor_linear_final:,.2f}
        - **Diferença:** ${abs(valor_exp_final - valor_linear_final):,.2f}
        """)
    
    with tab5:
        st.subheader("Aplicações Econômicas do Crescimento Exponencial")
        
        aplicacao = st.selectbox(
            "Escolha uma Aplicação:",
            [
                "Crescimento Populacional",
                "Inflação Acumulada", 
                "Depreciação de Ativos",
                "Crescimento do PIB",
                "Juros Compostos vs Simples"
            ]
        )
        
        if aplicacao == "Crescimento Populacional":
            st.write("**Modelo:** P(t) = P₀ × e^(rt)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                pop_inicial = st.number_input("População Inicial:", value=1000000, step=10000, key="pop_ini")
                taxa_cresc_pop = st.number_input("Taxa de Crescimento (% a.a.):", value=2.0, key="taxa_pop") / 100
                anos_projecao = st.slider("Anos de Projeção:", 1, 100, 50, key="anos_pop")
            
            with col2:
                tempo_pop = np.linspace(0, anos_projecao, anos_projecao + 1)
                populacao = [crescimento_populacional(pop_inicial, taxa_cresc_pop, t) for t in tempo_pop]
                
                fig_pop = go.Figure()
                
                fig_pop.add_trace(go.Scatter(
                    x=tempo_pop,
                    y=populacao,
                    mode='lines+markers',
                    name='População',
                    line=dict(color='green', width=3)
                ))
                
                fig_pop.update_layout(
                    title=f"Projeção Populacional - Taxa: {taxa_cresc_pop:.1%}",
                    xaxis_title="Anos",
                    yaxis_title="População",
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig_pop, use_container_width=True)
            
            pop_final = crescimento_populacional(pop_inicial, taxa_cresc_pop, anos_projecao)
            crescimento_total = ((pop_final - pop_inicial) / pop_inicial) * 100
            
            st.success(f"""
            **Resultados da Projeção:**
            - **População Inicial:** {pop_inicial:,}
            - **População em {anos_projecao} anos:** {pop_final:,.0f}
            - **Crescimento Total:** {crescimento_total:.1f}%
            - **Tempo para Dobrar:** {np.log(2) / taxa_cresc_pop:.1f} anos
            """)
        
        elif aplicacao == "Juros Compostos vs Simples":
            st.write("**Comparação:** Juros Simples vs Compostos vs Contínuos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                capital_juros = st.number_input("Capital ($):", value=10000.0, step=1000.0, key="cap_juros")
                taxa_juros = st.number_input("Taxa (% a.a.):", value=12.0, key="taxa_juros_comp") / 100
                tempo_juros = st.slider("Tempo (anos):", 1, 30, 15, key="tempo_juros")
            
            with col2:
                tempo_j = np.linspace(0, tempo_juros, tempo_juros * 4)
                
                # Juros simples: M = C(1 + rt)
                valores_simples = [capital_juros * (1 + taxa_juros * t) for t in tempo_j]
                
                # Juros compostos: M = C(1 + r)^t
                valores_compostos = [valor_futuro_discreto(capital_juros, taxa_juros, t) for t in tempo_j]
                
                # Juros contínuos: M = Ce^(rt)
                valores_continuos = [valor_futuro_continuo(capital_juros, taxa_juros, t) for t in tempo_j]
                
                fig_juros = go.Figure()
                
                fig_juros.add_trace(go.Scatter(
                    x=tempo_j, y=valores_simples,
                    mode='lines', name='Juros Simples',
                    line=dict(color='blue', width=2)
                ))
                
                fig_juros.add_trace(go.Scatter(
                    x=tempo_j, y=valores_compostos,
                    mode='lines', name='Juros Compostos',
                    line=dict(color='red', width=2)
                ))
                
                fig_juros.add_trace(go.Scatter(
                    x=tempo_j, y=valores_continuos,
                    mode='lines', name='Juros Contínuos',
                    line=dict(color='green', width=2, dash='dash')
                ))
                
                fig_juros.update_layout(
                    title="Comparação: Juros Simples vs Compostos vs Contínuos",
                    xaxis_title="Tempo (anos)",
                    yaxis_title="Montante ($)",
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig_juros, use_container_width=True)
            
            # Comparação final
            montante_simples = capital_juros * (1 + taxa_juros * tempo_juros)
            montante_composto = valor_futuro_discreto(capital_juros, taxa_juros, tempo_juros)
            montante_continuo = valor_futuro_continuo(capital_juros, taxa_juros, tempo_juros)
            
            st.subheader("Comparação Final")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Juros Simples", f"${montante_simples:,.2f}")
            
            with col2:
                diferenca_comp = montante_composto - montante_simples
                st.metric("Juros Compostos", f"${montante_composto:,.2f}", f"+${diferenca_comp:,.2f}")
            
            with col3:
                diferenca_cont = montante_continuo - montante_simples
                st.metric("Juros Contínuos", f"${montante_continuo:,.2f}", f"+${diferenca_cont:,.2f}")
    
    # Seção educativa
    with st.expander("Conceitos Importantes"):
        st.markdown("""
        ### O Número e na Economia
        
        **Definição:** e ≈ 2.71828... é definido como lim(m→∞) (1 + 1/m)^m
        
        **Interpretação Econômica:** 
        - Valor final de $1 investido a 100% ao ano com capitalização contínua
        - Base natural para crescimento exponencial contínuo
        
        ### Equações
        
        **Capitalização Discreta:** V = A(1 + i)^t
        - i = taxa efetiva por período
        - t = número de períodos
        
        **Capitalização Contínua:** V = Ae^(rt)
        - r = taxa contínua
        - t = tempo contínuo
        
        **Conversão de Taxas:**
        - Discreta → Contínua: r = ln(1 + i)
        - Contínua → Discreta: i = e^r - 1
        
        ### Taxa Instantânea de Crescimento
        
        Para V = Ae^(rt):
        - dV/dt = rAe^(rt) = rV
        - Taxa de crescimento = r (constante)
        
        ### Aplicações Práticas
        
        - **Finanças:** Capitalização contínua, precificação de derivativos
        - **Demografia:** Crescimento populacional
        - **Economia:** Crescimento do PIB, inflação
        - **Física:** Decaimento radioativo, depreciação
        
        ### Vantagens da Capitalização Contínua
        
        - Simplicidade matemática em cálculos avançados
        - Base para modelos estocásticos
        - Facilita análises de sensibilidade
        - Importante para Black-Scholes e outros modelos
        """)
    
    st.info("""
    **Nota Pedagógica:** O número e não é apenas uma constante matemática, mas representa 
    um conceito importante em finanças: o limite natural do crescimento exponencial. 
    Compreender sua interpretação econômica é essencial para modelos financeiros avançados.
    """)