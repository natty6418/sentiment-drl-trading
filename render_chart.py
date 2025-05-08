# render_chart.py
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Load data
gating_account_value = pd.read_csv('data/gating_account_value.csv')
baseline_df = pd.read_csv('data/baseline_df.csv')

# Make sure 'date' column is datetime
gating_account_value['date'] = pd.to_datetime(gating_account_value['date'])

# Paste your full visualization code here,
# replacing the df and baseline_df assignment lines with the ones above
# (you can reuse the exact code I gave you before)
# 1. CHOOSE YOUR RENDERER
pio.renderers.default = "browser"

# 2. LOAD & PREPARE YOUR DATA
df = gating_account_value.copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

df['Open']  = baseline_df['open']
df['High']  = baseline_df['high']
df['Low']   = baseline_df['low']
df['Close'] = baseline_df['close']

# 3. BUILD THE FRAMES (each frame shows 1 candle + line chart)
frames = []

for k in range(1, len(df) + 1):
    slice_ = df.iloc[:k]

    candles = go.Candlestick(
        x=slice_['date'],
        open=slice_['Open'],
        high=slice_['High'],
        low=slice_['Low'],
        close=slice_['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        increasing_line_width=6,  # <-- wider candles
        decreasing_line_width=6,
        showlegend=False,
        yaxis='y2'
    )

    account_trace = go.Scatter(
        x=slice_['date'],
        y=slice_['account_value'],
        mode='lines',
        name='Account Value',
        line=dict(width=2, color='cyan')
    )

    val = slice_['account_value'].iloc[-1]
    date_str = slice_['date'].iloc[-1].strftime('%Y-%m-%d')
    annot = go.layout.Annotation(
        text=f"Date: {date_str}<br>Account Value: ${val:,.2f}",
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        showarrow=False,
        font=dict(color='white', size=18),
        bgcolor="rgba(0,0,0,0.5)"
    )

    frames.append(go.Frame(data=[candles, account_trace], name=str(k), layout=go.Layout(annotations=[annot])))

# 4. INITIAL FIGURE
fig = go.Figure(frames=frames)

# Add initial traces
fig.add_trace(go.Candlestick(
    x=df['date'].iloc[:1],
    open=df['Open'].iloc[:1],
    high=df['High'].iloc[:1],
    low=df['Low'].iloc[:1],
    close=df['Close'].iloc[:1],
    increasing_line_color='green',
    decreasing_line_color='red',
    increasing_line_width=6,
    decreasing_line_width=6,
    showlegend=False,
    yaxis='y2'
))

fig.add_trace(go.Scatter(
    x=df['date'].iloc[:1],
    y=df['account_value'].iloc[:1],
    mode='lines',
    name='Account Value',
    line=dict(width=2, color='cyan')
))

# Initial annotation
init_date = df['date'].iloc[0].strftime('%Y-%m-%d')
init_val  = df['account_value'].iloc[0]
fig.add_annotation(
    text=f"Date: {init_date}<br>Account Value: ${init_val:,.2f}",
    x=0.02, y=0.98,
    xref='paper', yref='paper',
    showarrow=False,
    font=dict(color='white', size=18),
    bgcolor="rgba(0,0,0,0.5)",
)

# 5. ANIMATION + LAYOUT
anim_args = {
    "frame": {"duration": 2000, "redraw": True},
    "fromcurrent": True,
    "transition": {"duration": 100}
}

fig.update_layout(
    font=dict(color="white", size=18),
    template="plotly_dark",
    paper_bgcolor="black",
    plot_bgcolor="black",
    autosize=True,
    margin=dict(l=0, r=0, t=40, b=0),
    title="Account Value + Candlestick Chart (Wider Candles)",
    title_x=0.5,

    # Axes Config
    xaxis=dict(
        showgrid=False,
        range=[df['date'].min(), df['date'].max()]
    ),
    yaxis=dict(
        title='Account Value',
        showgrid=False,
        range=[
            df['account_value'].min() * 0.98,
            df['account_value'].max() * 1.02
        ]
    ),
    yaxis2=dict(
        title='Price',
        overlaying='y',
        side='right',
        showgrid=False,
        range=[
            df['Low'].min() * 0.98,
            df['High'].max() * 1.02
        ]
    ),

    xaxis_rangeslider_visible=False,

    # Animation controls
    updatemenus=[{
        "buttons": [
            {"args": [None, anim_args], "label": "Play",  "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
             "label": "Pause", "method": "animate"}
        ],
        "type": "buttons", "showactive": False,
        "x": 0.1, "y": 0, "pad": {"t": 60}
    }],
    sliders=[{
        "active": 0,
        "currentvalue": {"prefix": "Frame: ", "visible": True},
        "pad": {"b": 10, "t": 50},
        "len": 0.9, "x": 0.05, "y": 0,
        "steps": [
            {"args": [[f.name], anim_args], "label": f.name, "method": "animate"}
            for f in frames
        ]
    }]
)

# 6. SHOW FULL-SCREEN & RESPONSIVE
fig.show(config={"responsive": True, "displayModeBar": True})

# At the end of the file:
pio.write_html(fig, file='chart.html', auto_open=True, full_html=True, include_plotlyjs='cdn')
