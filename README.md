# Ferramentas Quantitativas - Dashboard Educacional de Finanças

Um conjunto abrangente de ferramentas interativas desenvolvidas em Streamlit para ensino e aplicação de conceitos quantitativos em finanças, incluindo precificação de opções, análise de risco e estrutura de capital.

## Descrição

Este aplicativo educacional oferece cinco módulos principais para análise financeira quantitativa, permitindo aos usuários explorar conceitos teóricos através de implementações práticas e visualizações interativas. Ideal para estudantes, professores e profissionais de finanças.

## Ferramentas Disponíveis

O dashboard inclui calculadoras e simuladores para:

- Precificação de opções (Black-Scholes-Merton)
- Análise de sensibilidade (Gregas)
- Visualização de payoffs
- Simulações Monte Carlo
- Análise de estrutura de capital (WACC, CAPM)

## Requisitos

```
streamlit
numpy
plotly
scipy
pandas
```

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd ferramentas-quantitativas
```

2. Instale as dependências:
```bash
pip install streamlit numpy plotly scipy pandas
```

## Como Usar

1. Execute o aplicativo:
```bash
streamlit run app.py
```

2. Acesse o dashboard no navegador (geralmente `http://localhost:8501`)

3. Use o menu lateral para navegar entre as diferentes ferramentas

4. Configure os parâmetros e analise os resultados em tempo real

## Uso Educacional

Este projeto foi desenvolvido especificamente para fins educacionais, oferecendo:

- Interface intuitiva para exploração de conceitos financeiros
- Visualizações interativas para melhor compreensão
- Implementações fiéis aos modelos teóricos
- Glossários e explicações dos conceitos utilizados

## Autor

**Prof. Vinicio Almeida**  
LinkedIn: https://linkedin.com/in/vinicioalmeida/  
Email: almeida.vinicio@gmail.com

## Estrutura do Projeto

```
├── app.py                 # Arquivo principal do aplicativo
├── README.md             # Este arquivo
└── requirements.txt      # Dependências do projeto
```

## Observações Técnicas

- Implementação baseada em modelos teóricos clássicos
- Uso de bibliotecas científicas para precisão numérica
- Interface responsiva e intuitiva
- Visualizações interativas com Plotly

## Aviso Importante

Este projeto é destinado exclusivamente para fins educacionais e acadêmicos. As ferramentas fornecidas são baseadas em modelos teóricos e não devem ser utilizadas como única fonte para decisões de investimento real.

## Licença

Este projeto é de uso acadêmico e educacional.