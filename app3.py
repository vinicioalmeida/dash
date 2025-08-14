import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import pandas as pd

# Configuração da página
st.set_page_config(page_title='Ferramentas Quantitativas')

st.markdown('<span style="color:gold; font-size: 48px">&#9733;</span> <span style="font-size: 48px; font-weight: bold">Ferramentas Quantitativas</span>', unsafe_allow_html=True)
st.markdown("""Escolha à esquerda a ferramenta (no celular, seta bem em cima à esquerda).""")
st.markdown('---')

# Configuração da barra lateral
st.sidebar.markdown('---')
selected_tool = st.sidebar.radio(
    "Escolha a Ferramenta:",
    ["Calculadora de VPL", "Estrutura de Capital", "Calculadoras Black-Scholes-Merton", "Calculadora de Gregas de Opções", "Payoff de Opções", "Simulador de Monte Carlo"]
)

st.sidebar.markdown("[Simulador cambial](https://simuladorcambio.streamlit.app)")

st.sidebar.markdown('---')
st.sidebar.markdown("""
    Prof. Vinicio Almeida \\
    https://linkedin.com/in/vinicioalmeida/ \\
    almeida.vinicio@gmail.com
    """)

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

# Implementação das funcionalidades
if selected_tool == "Calculadora de VPL":
    st.subheader('Calculadora de Valor Presente Líquido (VPL)')
    
    st.write("""
    O **Valor Presente Líquido (VPL)** é uma ferramenta fundamental para avaliação de projetos de investimento.
    
    **Fórmula:** VPL = C₀ + Σ[Cₜ/(1+r)ᵗ]
    
    Onde:
    - C₀ = Investimento inicial (fluxo negativo)
    - Cₜ = Fluxo de caixa no período t
    - r = Taxa de desconto (custo de capital)
    - t = Período
    """)
    
    # Tabs para diferentes tipos de análise
    tab1, tab2, tab3 = st.tabs(["📊 Cálculo Básico", "📈 Análise de Sensibilidade", "🎯 Múltiplos Projetos"])
    
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
                ✅ **PROJETO VIÁVEL**
                
                O VPL positivo de R$ {vpl:.2f} mil indica que o projeto:
                - Gera valor para a empresa
                - Tem retorno superior ao custo de capital ({taxa_desconto*100:.1f}%)
                - Deve ser **ACEITO**
                """)
            elif vpl < 0:
                st.error(f"""
                ❌ **PROJETO NÃO VIÁVEL**
                
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
                🏆 **Recomendação:** {projeto_melhor['Projeto']}
                
                - Maior VPL: R$ {projeto_melhor['VPL (R$ mil)']:.2f} mil
                - TIR: {projeto_melhor['TIR (%)']}%
                - Índice de Lucratividade: {projeto_melhor['IL']}
                """)
            else:
                st.warning("⚠️ **Nenhum projeto apresenta VPL positivo.** Considere revisar os parâmetros ou buscar alternativas.")
    
    # Seção educativa
    with st.expander("📚 Conceitos Importantes sobre VPL"):
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

elif selected_tool == "Calculadoras Black-Scholes-Merton":
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

elif selected_tool == "Calculadora de Gregas de Opções":
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

elif selected_tool == "Estrutura de Capital":
    st.subheader('Análise de Estrutura de Capital')
    
    st.write("""
    Esta ferramenta calcula o **WACC** (Custo Médio Ponderado de Capital), **CAPM** (Modelo de Precificação de Ativos Financeiros), 
    e analisa a estrutura de capital ótima considerando benefícios fiscais e custos de falência.
    """)
    
    # Abas para diferentes análises
    tab1, tab2, tab3 = st.tabs(["📊 Calculadora WACC", "📈 Análise CAPM", "⚖️ Estrutura Ótima"])
    
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
                st.write("**📊 Análise dos Resultados:**")
                st.write(f"• **Estrutura ótima:** {estrutura_otima['Debt_Ratio']:.1%} de endividamento")
                st.write(f"• **WACC mínimo:** {estrutura_otima['WACC']:.2%}")
                st.write(f"• **Economia fiscal anual:** R$ {estrutura_otima['Tax_Shield']:.0f} milhões")
                st.write(f"• **Aumento de valor:** R$ {max_valor_empresa['Valor_Empresa'] - pl_otimo:.0f} milhões")
                
            with col2:
                st.write("**⚠️ Considerações Importantes:**")
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
                    st.success(f"💡 **Recomendação:** A empresa pode criar R$ {diferenca_valor:.0f} milhões em valor ajustando sua estrutura de capital para o nível ótimo.")
                elif diferenca_valor < -50:  # Tolerância de R$ 50M
                    st.warning("⚠️ **Atenção:** A empresa pode estar super-endividada. Considere reduzir o endividamento.")
                else:
                    st.info("✅ **Status:** A estrutura atual está próxima do ótimo teórico.")
    
    # Seção de glossário e explicações
    with st.expander("📚 Glossário e Conceitos"):
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
    **⚠️ Importante:** Esta ferramenta fornece estimativas baseadas em modelos teóricos. 
    Para decisões importantes de estrutura de capital, considere:
    - Análise detalhada do setor e concorrentes
    - Condições específicas da empresa
    - Flexibilidade financeira e acesso ao mercado
    - Consultoria com especialistas em finanças corporativas
    """)