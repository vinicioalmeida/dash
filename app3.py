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
    ["Calculadoras Black-Scholes-Merton", "Calculadora de Gregas de Opções", "Payoff de Opções", "Simulador de Monte Carlo"]
)

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

# Implementação das funcionalidades
if selected_tool == "Calculadoras Black-Scholes-Merton":
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