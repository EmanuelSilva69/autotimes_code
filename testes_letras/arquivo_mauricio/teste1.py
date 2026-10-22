import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL

# 1. Carregar e tratar os dados
df = pd.read_csv("E:/2026.1/Viplab/AutoTimes/dataset/ETT-small/ETTh1.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('h')

# 2. Ajustar o STL
res = STL(df['OT'], period=7).fit()

# 3. Criar subplots interativos
fig = make_subplots(rows=4, cols=1, subplot_titles=('Observed', 'Trend', 'Seasonal', 'Resid'))

fig.add_trace(go.Scatter(x=res.observed.index, y=res.observed, name='Observed'), row=1, col=1)
fig.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name='Trend'), row=2, col=1)
fig.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, name='Seasonal'), row=3, col=1)
fig.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name='Resid'), row=4, col=1)

fig.update_layout(height=1000, title_text="Decomposição STL (Interativa)", showlegend=False)
fig.show()