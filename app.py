import plotly
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import datetime
from datetime import datetime, timedelta, time
import math
from scipy import stats
from scipy.stats import norm
from scipy.optimize import newton
from openpyxl import Workbook, load_workbook
import requests
import zipfile
import io
import fundamentus
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Configuração da página
st.set_page_config(page_title='Ferramentas Quantitativas')

st.markdown('<span style="color:gold; font-size: 48px">&#9733;</span> <span style="font-size: 48px; font-weight: bold">Ferramentas Quantitativas</span>', unsafe_allow_html=True)
st.markdown("""Escolha à esquerda a ferramenta (no celular, seta bem em cima à esquerda).""")
st.markdown('---')

st.sidebar.markdown('---')
selected_calculator = st.sidebar.radio(
    "Ferramentas:",
    ("Calculadoras Black-Scholes-Merton", "Calculadora de Gregas de Opções")
)

st.sidebar.markdown('---')
st.sidebar.markdown("""
    Prof. Vinicio Almeida \\
    https://linkedin.com/in/vinicioalmeida/ \\
    almeida.vinicio@gmail.com
    """)


###########################
### BLACK-SCHOLES

if selected_calculator == "Calculadoras Black-Scholes-Merton":
    
    ### Calculadora Black-Scholes-Merton - Preço
    def black_scholes_call_put(S, K, T, r, sigma, option_type):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == 'call':
            option_price = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        elif option_type == 'put':
            option_price = K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        return option_price

    # Função de distribuição cumulativa normal padrão (CDF)
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    # Título do aplicativo
    st.subheader('Calculadora Black-Scholes-Merton - Preço da opção')
    st.markdown("""
    Black, Scholes e Merton revolucionaram a análise de opções, fornecendo um arcabouço robusto 
                para entender e precificar riscos financeiros. Sua influência perdura, moldando 
                a maneira como investidores e instituições lidam com a complexidade dos mercados. 
    """)
    st.markdown('---')
    # Organizando os campos de entrada em duas colunas
    col1, col2 = st.columns(2)

    # Entrada dos valores dos parâmetros
    with col1:
        S = st.number_input('Preço do ativo subjacente (S)', min_value=0.0)
        K = st.number_input('Preço de exercício da opção (K)', min_value=0.0)
        T = st.number_input('Vencimento da opção (T), como fração do ano', min_value=0.0)
    with col2:
        r = st.number_input('Taxa de juros anual (r), como fração', min_value=0.0)
        sigma = st.number_input('Volatilidade (Sigma), como fração', min_value=0.0)
        option_type = st.selectbox('Tipo de opção', ['call', 'put'])

    # Botão para calcular o preço da opção
    if st.button('Calcular preço da opção'):
        option_price = black_scholes_call_put(S, K, T, r, sigma, option_type)
        st.write(f'O preço da {option_type} é: ${round(option_price,2)}')

    st.markdown('---')

    # Função para calcular o preço da opção Black-Scholes-Merton
    def black_scholes_call_put(S, K, T, r, sigma, option_type):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == 'call':
            option_price = S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
        elif option_type == 'put':
            option_price = K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        return option_price

    # Função para calcular a volatilidade implícita
    def implied_volatility(option_price, S, K, T, r, option_type):
        # Função para encontrar a volatilidade implícita usando o método de Newton-Raphson
        def func(sigma, option_price, S, K, T, r, option_type):
            return black_scholes_call_put(S, K, T, r, sigma, option_type) - option_price

        # Chamada para o método de Newton-Raphson para encontrar a volatilidade implícita
        sigma = newton(func, x0=0.2, args=(option_price, S, K, T, r, option_type))
        return sigma

    # Título do aplicativo
    st.subheader('Calculadora Black-Scholes-Merton - Volatilidade implícita')
    st.markdown("""
    A volatilidade implícita, elemento-chave no modelo Black-Scholes, reflete as expectativas 
                do mercado sobre a flutuação futura dos preços de ativos. Sua análise é crucial 
                para precificar opções e compreender as percepções dos investidores em relação 
                ao risco. 
    """)
    # Organizando os campos de entrada em duas colunas
    col1, col2 = st.columns(2)

    # Entrada dos valores dos parâmetros
    with col1:
        Sv = st.number_input('Preço do ativo subjacente (S)', min_value=0.0, key='Sv')
        Kv = st.number_input('Preço de exercício da opção (K)', min_value=0.0, key='Kv')
        Tv = st.number_input('Vencimento da opção (T), como fração do ano', min_value=0.0, key='Tv')
    with col2:
        rv = st.number_input('Taxa de juros anual (r), como fração', min_value=0.0, key='rv')
        option_price = st.number_input('Preço da opção', min_value=0.0, key='option_price')
        option_type = st.selectbox('Tipo de opção', ['call', 'put'], key='option_type')

    # Botão para calcular a volatilidade implícita
    if st.button('Calcular volatilidade implícita'):
        sigma_impl = implied_volatility(option_price, Sv, Kv, Tv, rv, option_type)
        st.write(f'A volatilidade implícita é {round(sigma_impl,2)}')

###########################
### GREGAS

elif selected_calculator == "Calculadora de Gregas de Opções":
    # Calculadora de gregas
    # Função para calcular a grega Delta
    def delta(S, K, T, r, sigma, option_type):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

        if option_type == 'call':
            delta = norm.cdf(d1)
        elif option_type == 'put':
            delta = norm.cdf(d1) - 1
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        return delta

    # Função para calcular a grega Gamma
    def gamma(S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
        return gamma

    # Função para calcular a grega Vega
    def vega(S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        return vega

    # Função para calcular a grega Theta
    def theta(S, K, T, r, sigma, option_type):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == 'call':
            theta = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            theta = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        return theta

    # Função para calcular a grega Rho
    def rho(S, K, T, r, sigma, option_type):
        d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

        if option_type == 'call':
            rho = K * T * math.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        return rho

    # Título do aplicativo
    st.subheader('Calculadora de Gregas de Opções')
    st.markdown("""
        As 'Gregas', na análise de derivativos, referem-se a medidas de sensibilidade de preço 
                de opções a diferentes variáveis, como volatilidade, tempo e  movimento do preço 
                do ativo subjacente. Delta, Gamma, Vega, Theta e Rho são alguns exemplos essenciais. 
        """)
    st.markdown('---')
    # Organizando os campos de entrada em duas colunas
    col1, col2 = st.columns(2)

    # Entrada dos valores dos parâmetros
    with col1:
        S = st.number_input('Preço do ativo subjacente (S)', min_value=0.0, step=0.01, key='S_gregas')
        K = st.number_input('Preço de exercício da opção (K)', min_value=0.0, step=0.01, key='K_gregas')
        T = st.number_input('Tempo até o vencimento (T) em anos', min_value=0.0, step=0.01, key='T_gregas')
    with col2:
        r = st.number_input('Taxa de juros anual (r), como fração', min_value=0.0, step=0.0001, key='r_gregas')
        sigma = st.number_input('Volatilidade (Sigma), como fração', min_value=0.0, step=0.0001, key='sigma_gregas')
        option_type = st.selectbox('Tipo de opção', ['call', 'put'], key='option_type_gregas')

    # Botão para calcular as "gregas"
    if st.button('Calcular Gregas'):
        st.write("Delta:", round(delta(S, K, T, r, sigma, option_type),6))
        st.write("Gamma:", round(gamma(S, K, T, r, sigma),6))
        st.write("Vega:", round(vega(S, K, T, r, sigma),6))
        st.write("Theta:", round(theta(S, K, T, r, sigma, option_type),6))
        st.write("Rho:", round(rho(S, K, T, r, sigma, option_type),6))
