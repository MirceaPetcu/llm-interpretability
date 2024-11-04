import plotly.graph_objects as go

# Datele furnizate
layers = list(range(33))

correlations = [
    0.28048094376031174, 0.32077135128607875, 0.3313165908911906, 0.34696588191816263, 0.3674517528974853,
    0.3826834851445901, 0.3796515238493794, 0.3884295587416625, 0.3862706678615432, 0.40082372535048744,
    0.41390965520906536, 0.42693989912309266, 0.4398025010300227, 0.4410222123020976, 0.45486635551583754,
    0.452157720279612, 0.4529676938444984, 0.45346539959554805, 0.44779121003589023, 0.437563172745892,
    0.4311707513529397, 0.4139541436171165, 0.4084019538590228, 0.40765031066095264, 0.40008727362587493,
    0.40026988018661225, 0.40927287484369684, 0.40623465223357824, 0.4114465789997073, 0.39453997153411435,
    0.4039844456917635, 0.3791955218650675, 0.372209817528124
]

# Creează graficul
fig = go.Figure()

# Adaugă traseul de date cu legenda "Mixtral"
fig.add_trace(go.Scatter(
    x=layers,
    y=correlations,
    mode='lines+markers',
    name='Red - Mixtral 7x8B 4bit',  # Numele legendei
    line=dict(color='red')
))

# Actualizează layout-ul
fig.update_layout(
    title='Pearson Correlation by Layer',
    xaxis_title='Layer',
    yaxis_title='Pearson Correlation',
    xaxis=dict(
        tickvals=list(range(41)),  # Afișează numerele întregi de la 0 la 40
        dtick=1  # Intervalul dintre tichete
    ),
    yaxis=dict(
        tickvals=[i / 20 for i in range(11)],  # Afișează numerele de la 0.0 la 1.0
        dtick=0.01  # Intervalul dintre tichete
    ),
    plot_bgcolor='black',  # Fundalul graficului
    paper_bgcolor='black',  # Fundalul întregului plot
    font=dict(color='white'),  # Schimbă culoarea textului
    showlegend=True,
    legend=dict(
        x=1,
        y=1,
        xanchor='right',  # Ancorare la dreapta
        yanchor='top',
        bgcolor="darkblue",  # Fundalul legendei negru
        bordercolor='blue',  # Bordură albă pentru legendă
        borderwidth=10,  # Lățimea bordurii
        font=dict(color='red')
    )
)

# Afișează graficul
fig.show()
