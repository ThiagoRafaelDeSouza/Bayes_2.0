from flask import Flask, render_template
from dash import Dash, dcc, html, Input, Output
import numpy as np
import plotly.graph_objects as go
from scipy.stats import beta, binom, gamma

# Funções de probabilidade (mesmo que antes)

def plot_beta_distribution(a, b):
    theta = np.linspace(0, 1, 1000)
    if a == 1 and b == 1:
        beta_pdf = np.ones_like(theta)
    else:
        beta_pdf = beta.pdf(theta, a, b)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theta,
        y=beta_pdf,
        mode='lines',
        name=f'Beta({a}, {b})',
        line=dict(color='royalblue', width=2)
    ))

    fig.update_layout(
        title=f'Distribuição Beta({a}, {b})',
        xaxis_title='x',
        yaxis_title='Densidade',
        template='plotly_white'
    )
    return fig

def plot_gamma_distribution(a, b):
    if a > 1:
        mode = (a - 1) / b
    else:
        mode = 0
    x_min = max(0, mode - 10 / b)
    x_max = mode + 10 / b
    x = np.linspace(x_min, x_max, 1000)
    gamma_pdf = gamma.pdf(x, a, scale=1/b)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=gamma_pdf,
        mode='lines',
        name=f'Gamma({a}, {b})',
        line=dict(color='royalblue', width=2)
    ))
    fig.update_layout(
        title=f'Distribuição Gama({a}, {b})',
        xaxis_title='x',
        yaxis_title='Densidade',
        template='plotly_white',
        showlegend=True
    )
    return fig

def plot_binomial_distribution(n, p):
    x = np.arange(0, n + 1)
    binomial_pmf = binom.pmf(x, n, p)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=binomial_pmf,
        mode='markers',
        name=f'Binomial({n}, {p})',
        marker=dict(color='royalblue', size=10)
    ))
    fig.update_layout(
        title=f'Binomial Distribution (n={n}, p={p})',
        xaxis_title='Número de sucessos (x)',
        yaxis_title='Probabilidade',
        template='plotly_white'
    )
    return fig

def plot_bernoulli_likelihood(sample_size, sample_mean):
    p_values = np.linspace(0, 1, 1000)
    likelihood_values = p_values**(sample_mean * sample_size) * (1 - p_values)**((1 - sample_mean) * sample_size)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p_values,
        y=likelihood_values,
        mode='lines',
        name='Verossimilhança',
        line=dict(color='blue')
    ))
    fig.update_layout(
        title=f'Função de Verossimilhança da Bernoulli (n = {sample_size}, média = {sample_mean})',
        xaxis_title='p (Probabilidade de Sucesso)',
        yaxis_title='Verossimilhança',
        template='plotly_white'
    )
    return fig

# Criação da aplicação Flask
server = Flask(__name__)

# Criação da aplicação Dash com integração ao Flask
app = Dash(__name__, server=server, url_base_pathname='/dashboard/')

lista_prioris = list(["Beta", "Gamma"])

app.layout = html.Div(children=[
    html.H1(children='Trabalho de Bayesiana - Grupo PET Estatística'),
    html.H4('Selecione a distribuição da priori'),
    dcc.Dropdown(lista_prioris, id='prioris'),
    html.H4('Selecione a distribuição da verossimilhança'),
    dcc.Dropdown(id='verossimilhancas'),
    html.H4('Digite os parâmetros da priori'),
    html.Label(id="texto_priori_1"),
    dcc.Input(id='input-a', type='number'),
    html.Label(id="texto_priori_2"),
    dcc.Input(id='input-b', type='number'),
    html.Br(),
    dcc.Graph(id='densidade_priori'),
    html.H4('Digite os parâmetros da verossimilhança'),
    html.Label(id="texto_verossimilhanca_1"),
    dcc.Input(id='input-m', type='number'),
    html.Label(id='texto_posteriori_1'),
    dcc.Input(id='input-x', type='number', step=0.01),
    html.Label(children="Digite o tamanho amostral:", id='texto_posteriori_2'),
    dcc.Input(id='input-tamanho', type='number'),
    html.Br(),
    html.Div(
        dcc.Graph(id='densidade_verossimilhanca'),
        id='aparencia_verossimilhanca'
    ),
    html.Br(),
    dcc.Graph(id="densidade_posteriori")
])

# As callbacks continuam as mesmas...

@app.callback(
    Output("verossimilhancas", "options"),
    Input("prioris", "value")
)
def update_dropdown(prioris):
    if prioris == "Beta":
        return list(["Bernoulli", "Binomial"])
    elif prioris == "Gamma":
        return list(["Poisson", "Exponencial"])
    else:
        return list(["Outra"])

@app.callback(
    Output("texto_priori_1", "children"),
    Input("prioris", "value")
)
def update_dropdown(prioris):
    if prioris == "Beta" or prioris == "Gamma":
        return "Digite o valor de a:"
    else:
        return "Digite o valor de lambda:"

@app.callback(
    [Output("texto_priori_2", "children"),
     Output("input-b", "style")],
    Input("prioris", "value")
)
def update_dropdown(prioris):
    if prioris == "Beta" or prioris == "Gamma":
        return "Digite o valor de b:", {}
    else:
        return "", {"display": "none"}

@app.callback(
    Output('densidade_priori', 'figure'),
    [Input('input-a', 'value'),
     Input('input-b', 'value'),
     Input("prioris", "value")]
)
def update_output(a, b, prioris):
    if prioris == "Beta":
        return plot_beta_distribution(a, b)
    else:
        return plot_gamma_distribution(a, b)

# Continue com as outras callbacks...
# Rota principal do Flask
@server.route('/')
def index():
    return render_template('index.html')

# Run do servidor Flask
if __name__ == '__main__':
    server.run(debug=True)
