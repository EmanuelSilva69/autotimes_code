import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import MSTL

# 1. Carregar e tratar os dados
df = pd.read_csv("E:/2026.1/Viplab/AutoTimes/dataset/ETT-small/ETTh1.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('h')

# 2. Definir períodos e ajustar o MSTL
periods = (6,12, 24,49, 64, 256, 360)
res = MSTL(df['OT'], periods=periods).fit()

# 3. Número total de linhas = Observed (1) + Trend (1) + len(periods) + Resid (1)
num_rows = 2 + len(periods) + 1

titles = ['Observed', 'Trend'] + [f'Seasonal ({p}h)' for p in periods] + ['Resid']

fig = make_subplots(rows=num_rows, cols=1, subplot_titles=titles, shared_xaxes=True)

# Observed (Linha 1)
fig.add_trace(go.Scatter(x=res.observed.index, y=res.observed, name='Observed'), row=1, col=1)

# Trend (Linha 2)
fig.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name='Trend'), row=2, col=1)

# Componentes sazonais (Linhas 3 até 3 + len(periods) - 1)
seasonal_cols = res.seasonal.columns
for i, col in enumerate(seasonal_cols, start=3):
    fig.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal[col], name=f'Seasonal ({col})'), row=i, col=1)

# Resíduo (Última linha)
resid_row = num_rows
fig.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name='Resid'), row=resid_row, col=1)

# Ajuste de layout
fig.update_layout(
    height=250 * num_rows,
    title_text=f"Decomposição MSTL com {len(periods)} Sazonalidades (Interativa)",
    showlegend=False
)

fig.show()