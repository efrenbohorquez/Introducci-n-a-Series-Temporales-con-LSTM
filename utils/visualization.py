import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

def create_time_series_plot(df, x_col, y_cols, title, height=500, show_covid=True, covid_date='2020-01-01'):
    """
    Crea un gráfico de series temporales con Plotly.
    
    Args:
        df: DataFrame con los datos
        x_col: Nombre de la columna para el eje X (fecha)
        y_cols: Lista de nombres de columnas para el eje Y
        title: Título del gráfico
        height: Altura del gráfico en píxeles
        show_covid: Si se debe mostrar una línea vertical para COVID-19
        covid_date: Fecha de inicio de COVID-19 (formato YYYY-MM-DD)
    
    Returns:
        Figura de Plotly
    """
    # Crear figura
    fig = go.Figure()
    
    # Colores para las series
    colors = ['#1E88E5', '#26A69A', '#FF8F00', '#9C27B0', '#42A5F5']
    
    # Añadir cada serie temporal
    for i, y_col in enumerate(y_cols):
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            )
        )
    
    # Añadir línea vertical para COVID si está habilitado
    if show_covid:
        covid_date_dt = pd.to_datetime(covid_date)
        if (df[x_col].min() <= covid_date_dt) and (df[x_col].max() >= covid_date_dt):
            fig.add_shape(
                type="line",
                x0=covid_date_dt,
                x1=covid_date_dt,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dash", color="red", width=2)
            )
            fig.add_annotation(
                x=covid_date_dt,
                y=1,
                yref="paper",
                text="Inicio COVID-19",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                ax=20,
                ay=-30
            )
    
    # Actualizar diseño
    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Valor",
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_correlation_heatmap(df, cols=None, title="Matriz de Correlación"):
    """
    Crea un mapa de calor de correlación con Plotly.
    
    Args:
        df: DataFrame con los datos
        cols: Lista de columnas para incluir en la correlación (None para usar todas las numéricas)
        title: Título del gráfico
    
    Returns:
        Figura de Plotly
    """
    # Seleccionar columnas numéricas si no se especifican
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calcular matriz de correlación
    corr_matrix = df[cols].corr()
    
    # Crear mapa de calor
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title=title
    )
    
    fig.update_layout(
        height=600,
        width=700
    )
    
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None, title=None, add_trendline=True):
    """
    Crea un gráfico de dispersión con Plotly.
    
    Args:
        df: DataFrame con los datos
        x_col: Nombre de la columna para el eje X
        y_col: Nombre de la columna para el eje Y
        color_col: Nombre de la columna para colorear puntos (opcional)
        title: Título del gráfico (opcional)
        add_trendline: Si se debe añadir línea de tendencia
    
    Returns:
        Figura de Plotly
    """
    # Crear título si no se proporciona
    if title is None:
        title = f"Relación entre {y_col} y {x_col}"
    
    # Crear gráfico de dispersión
    if color_col:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            trendline="ols" if add_trendline else None,
            title=title,
            labels={x_col: x_col, y_col: y_col}
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            trendline="ols" if add_trendline else None,
            title=title,
            labels={x_col: x_col, y_col: y_col},
            color_discrete_sequence=['#1E88E5']
        )
    
    fig.update_traces(
        marker=dict(size=10, opacity=0.7, line=dict(width=1, color='black')),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        height=500,
        hovermode="closest"
    )
    
    return fig

def create_distribution_plot(df, col, title=None, bins=20, color='#1E88E5'):
    """
    Crea un histograma de distribución con Plotly.
    
    Args:
        df: DataFrame con los datos
        col: Nombre de la columna para visualizar
        title: Título del gráfico (opcional)
        bins: Número de bins para el histograma
        color: Color para el histograma
    
    Returns:
        Figura de Plotly
    """
    # Crear título si no se proporciona
    if title is None:
        title = f"Distribución de {col}"
    
    # Crear histograma
    fig = px.histogram(
        df, 
        x=col,
        nbins=bins,
        title=title,
        labels={col: col, "count": "Frecuencia"},
        color_discrete_sequence=[color]
    )
    
    # Añadir línea de densidad
    fig.update_layout(
        height=400,
        bargap=0.1
    )
    
    return fig

def create_comparison_plot(df1, df2, x_col, y_col, name1, name2, title=None):
    """
    Crea un gráfico comparativo entre dos DataFrames.
    
    Args:
        df1: Primer DataFrame
        df2: Segundo DataFrame
        x_col: Nombre de la columna para el eje X
        y_col: Nombre de la columna para el eje Y
        name1: Nombre para identificar el primer DataFrame
        name2: Nombre para identificar el segundo DataFrame
        title: Título del gráfico (opcional)
    
    Returns:
        Figura de Plotly
    """
    # Crear título si no se proporciona
    if title is None:
        title = f"Comparación de {y_col}"
    
    # Crear figura
    fig = go.Figure()
    
    # Añadir primera serie
    fig.add_trace(
        go.Scatter(
            x=df1[x_col],
            y=df1[y_col],
            mode='lines+markers',
            name=name1,
            line=dict(color='#1E88E5', width=2),
            marker=dict(size=8)
        )
    )
    
    # Añadir segunda serie
    fig.add_trace(
        go.Scatter(
            x=df2[x_col],
            y=df2[y_col],
            mode='lines+markers',
            name=name2,
            line=dict(color='#FF8F00', width=2),
            marker=dict(size=8)
        )
    )
    
    # Actualizar diseño
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500,
        hovermode="x unified"
    )
    
    return fig

def create_bar_chart(df, x_col, y_col, color_col=None, title=None, orientation='v'):
    """
    Crea un gráfico de barras con Plotly.
    
    Args:
        df: DataFrame con los datos
        x_col: Nombre de la columna para el eje X
        y_col: Nombre de la columna para el eje Y
        color_col: Nombre de la columna para colorear barras (opcional)
        title: Título del gráfico (opcional)
        orientation: Orientación del gráfico ('v' para vertical, 'h' para horizontal)
    
    Returns:
        Figura de Plotly
    """
    # Crear título si no se proporciona
    if title is None:
        title = f"{y_col} por {x_col}"
    
    # Crear gráfico de barras
    if color_col:
        fig = px.bar(
            df,
            x=x_col if orientation == 'v' else y_col,
            y=y_col if orientation == 'v' else x_col,
            color=color_col,
            title=title,
            orientation=orientation
        )
    else:
        fig = px.bar(
            df,
            x=x_col if orientation == 'v' else y_col,
            y=y_col if orientation == 'v' else x_col,
            title=title,
            orientation=orientation,
            color_discrete_sequence=['#1E88E5']
        )
    
    # Actualizar diseño
    fig.update_layout(
        height=500,
        xaxis_title=x_col if orientation == 'v' else y_col,
        yaxis_title=y_col if orientation == 'v' else x_col
    )
    
    return fig

def create_subplots(num_rows, num_cols, subplot_titles=None, shared_xaxes=True, vertical_spacing=0.1):
    """
    Crea una figura con subplots.
    
    Args:
        num_rows: Número de filas
        num_cols: Número de columnas
        subplot_titles: Lista de títulos para los subplots
        shared_xaxes: Si los ejes X deben compartirse
        vertical_spacing: Espaciado vertical entre subplots
    
    Returns:
        Figura de Plotly con subplots
    """
    fig = make_subplots(
        rows=num_rows, 
        cols=num_cols,
        shared_xaxes=shared_xaxes,
        vertical_spacing=vertical_spacing,
        subplot_titles=subplot_titles
    )
    
    return fig

def apply_custom_theme(fig):
    """
    Aplica un tema personalizado a una figura de Plotly.
    
    Args:
        fig: Figura de Plotly
    
    Returns:
        Figura de Plotly con tema aplicado
    """
    fig.update_layout(
        font=dict(family="Roboto, Arial, sans-serif", size=12),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        title_font=dict(size=20, color='#212529'),
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.1)',
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_xaxes(
        gridcolor='rgba(0, 0, 0, 0.1)',
        zerolinecolor='rgba(0, 0, 0, 0.2)',
        zerolinewidth=1.5
    )
    
    fig.update_yaxes(
        gridcolor='rgba(0, 0, 0, 0.1)',
        zerolinecolor='rgba(0, 0, 0, 0.2)',
        zerolinewidth=1.5
    )
    
    return fig

def format_large_number(num):
    """
    Formatea números grandes para mejor legibilidad.
    
    Args:
        num: Número a formatear
    
    Returns:
        Cadena formateada
    """
    if abs(num) >= 1e9:
        return f"{num/1e9:.2f} B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.2f} M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.2f} K"
    else:
        return f"{num:.2f}"

def add_annotations_to_plot(fig, annotations, row=1, col=1):
    """
    Añade anotaciones a un gráfico de Plotly.
    
    Args:
        fig: Figura de Plotly
        annotations: Lista de diccionarios con anotaciones
        row: Fila del subplot (para figuras con subplots)
        col: Columna del subplot (para figuras con subplots)
    
    Returns:
        Figura de Plotly con anotaciones
    """
    for annotation in annotations:
        fig.add_annotation(
            x=annotation.get('x'),
            y=annotation.get('y'),
            text=annotation.get('text', ''),
            showarrow=annotation.get('showarrow', True),
            arrowhead=annotation.get('arrowhead', 2),
            arrowsize=annotation.get('arrowsize', 1),
            arrowwidth=annotation.get('arrowwidth', 2),
            arrowcolor=annotation.get('arrowcolor', '#636363'),
            row=row,
            col=col
        )
    
    return fig

def create_gauge_chart(value, min_val, max_val, title, threshold_values=None, threshold_colors=None):
    """
    Crea un gráfico de indicador tipo gauge.
    
    Args:
        value: Valor a mostrar
        min_val: Valor mínimo del rango
        max_val: Valor máximo del rango
        title: Título del gráfico
        threshold_values: Lista de valores umbral para cambios de color
        threshold_colors: Lista de colores correspondientes a los umbrales
    
    Returns:
        Figura de Plotly
    """
    if threshold_values is None:
        threshold_values = [0.33, 0.66]
    
    if threshold_colors is None:
        threshold_colors = ['#FF4136', '#FFDC00', '#2ECC40']
    
    # Normalizar umbrales
    steps = []
    for i, threshold in enumerate(threshold_values + [1.0]):
        if i == 0:
            range_start = 0
        else:
            range_start = threshold_values[i-1]
        
        steps.append({
            'range': [range_start * (max_val - min_val) + min_val, threshold * (max_val - min_val) + min_val],
            'color': threshold_colors[i]
        })
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#1E88E5"},
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig
