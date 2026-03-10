import plotly.graph_objects as go
c_dict = {1: 10, 2: 5}
x_vals = sorted(c_dict.keys())
y_vals = [c_dict[x] for x in x_vals]
fig = go.Figure(data=[go.Bar(
    x=[str(x) for x in x_vals], 
    y=y_vals, 
    marker_color="red",
    text=y_vals,
    textposition='auto',
    hovertemplate="Title: %{x}次<br>数量: %{y}个<extra></extra>"
)])
fig.update_layout(
    margin=dict(l=20, r=20, t=10, b=20),
    xaxis=dict(type='category', title=""),
    yaxis=dict(visible=False),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=140
)
print("Plotly Figure created successfully.")
