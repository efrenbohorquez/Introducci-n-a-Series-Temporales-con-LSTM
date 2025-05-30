import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

def show(df_original, df_procesado):
    """
    Muestra la página de preprocesamiento de datos del tablero Streamlit.
    
    Args:
        df_original: DataFrame con los datos originales
        df_procesado: DataFrame con los datos procesados
    """
    st.markdown("## Preprocesamiento de Datos")
    
    # Crear pestañas para organizar el contenido
    tab1, tab2, tab3, tab4 = st.tabs(["Conversión de Fechas", "Valores Faltantes", "Detección de Outliers", "Transformaciones"])
    
    with tab1:
        st.markdown("### Conversión de Fechas")
        st.markdown("""
        El primer paso en el preprocesamiento de series temporales es asegurar que la variable temporal
        esté en el formato adecuado. En este caso, convertimos la columna 'TRIMESTRE' a formato datetime.
        """)
        
        # Mostrar código de conversión
        with st.expander("Ver código de conversión de fechas"):
            st.code("""
# Convertir trimestres a fechas (formato YYYY-Q)
def trimestre_a_fecha(trimestre):
    año, q = trimestre.split('-')
    mes = (int(q[-1]) - 1) * 3 + 1  # Q1->1, Q2->4, Q3->7, Q4->10
    return f"{año}-{mes:02d}-01"

# Aplicar conversión
df['fecha'] = df['TRIMESTRE'].apply(trimestre_a_fecha)
df['fecha'] = pd.to_datetime(df['fecha'])

# Establecer fecha como índice
df = df.set_index('fecha')
            """)
        
        # Mostrar resultado de la conversión
        st.markdown("#### Resultado de la Conversión")
        
        # Crear un DataFrame de ejemplo con la conversión
        df_ejemplo = df_original.copy()
        
        def trimestre_a_fecha(trimestre):
            año, q = trimestre.split('-')
            # Asegurar que q es un número válido
            q_num = int(q) if q.isdigit() else int(q[-1])
            mes = (q_num - 1) * 3 + 1
            return f"{año}-{mes:02d}-01"
        
        df_ejemplo['fecha'] = df_ejemplo['TRIMESTRE'].apply(trimestre_a_fecha)
        df_ejemplo['fecha'] = pd.to_datetime(df_ejemplo['fecha'])
        
        # Mostrar los primeros registros con la fecha convertida
        st.dataframe(df_ejemplo[['TRIMESTRE', 'fecha', 'IVA_TOTAL', 'PIB']].head(10), use_container_width=True)
        
        st.markdown("""
        La conversión a formato datetime permite:
        - Ordenar correctamente los datos cronológicamente
        - Utilizar funcionalidades de series temporales de pandas
        - Realizar resampling, rolling windows y otras operaciones temporales
        - Visualizar correctamente las series en gráficos temporales
        """)
    
    with tab2:
        st.markdown("### Detección y Tratamiento de Valores Faltantes")
        
        # Verificar valores faltantes en el dataset original
        missing_original = df_original.isnull().sum()
        
        st.markdown("#### Valores Faltantes en Datos Originales")
        
        if missing_original.sum() == 0:
            st.success("No se detectaron valores faltantes en el dataset original.")
        else:
            st.warning(f"Se detectaron {missing_original.sum()} valores faltantes en el dataset original.")
            st.dataframe(missing_original.to_frame('Valores Faltantes'), use_container_width=True)
        
        # Explicar métodos de imputación
        st.markdown("#### Métodos de Imputación para Series Temporales")
        
        st.markdown("""
        En caso de encontrar valores faltantes en series temporales, estos son algunos métodos comunes de imputación:
        
        1. **Interpolación Lineal**: Estima valores faltantes mediante una línea recta entre puntos conocidos.
        2. **Interpolación Polinómica**: Utiliza polinomios para estimar valores faltantes.
        3. **Imputación por Media/Mediana**: Reemplaza valores faltantes con la media o mediana de la serie.
        4. **Forward/Backward Fill**: Propaga el último/siguiente valor conocido.
        5. **Métodos basados en modelos**: Utiliza modelos como ARIMA para predecir valores faltantes.
        """)
        
        # Demostración de interpolación
        st.markdown("#### Demostración de Interpolación")
        
        # Crear un DataFrame de ejemplo con valores faltantes
        np.random.seed(42)
        df_demo = pd.DataFrame({
            'fecha': pd.date_range(start='2020-01-01', periods=12, freq='ME'),
            'valor': np.random.normal(100, 10, 12)
        })
        
        # Introducir valores faltantes
        df_demo.loc[[3, 7], 'valor'] = np.nan
        
        # Mostrar datos con valores faltantes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Datos con Valores Faltantes**")
            st.dataframe(df_demo, use_container_width=True)
        
        # Aplicar diferentes métodos de interpolación
        df_interp_linear = df_demo.copy()
        df_interp_linear['valor'] = df_interp_linear['valor'].interpolate(method='linear')
        
        df_interp_ffill = df_demo.copy()
        df_interp_ffill['valor'] = df_interp_ffill['valor'].ffill()
        
        with col2:
            metodo = st.selectbox(
                "Seleccione método de imputación:",
                ["Interpolación Lineal", "Forward Fill"]
            )
            
            if metodo == "Interpolación Lineal":
                st.markdown("**Datos con Interpolación Lineal**")
                st.dataframe(df_interp_linear, use_container_width=True)
            else:
                st.markdown("**Datos con Forward Fill**")
                st.dataframe(df_interp_ffill, use_container_width=True)
        
        # Visualización de la interpolación
        fig = go.Figure()
        
        # Datos originales con valores faltantes
        fig.add_trace(
            go.Scatter(
                x=df_demo['fecha'],
                y=df_demo['valor'],
                mode='markers+lines',
                name='Datos Originales',
                line=dict(color='#1E88E5', dash='dash'),
                marker=dict(size=10)
            )
        )
        
        # Datos con interpolación lineal
        fig.add_trace(
            go.Scatter(
                x=df_interp_linear['fecha'],
                y=df_interp_linear['valor'],
                mode='markers+lines',
                name='Interpolación Lineal',
                line=dict(color='#26A69A'),
                marker=dict(size=8)
            )
        )
        
        # Datos con forward fill
        fig.add_trace(
            go.Scatter(
                x=df_interp_ffill['fecha'],
                y=df_interp_ffill['valor'],
                mode='markers+lines',
                name='Forward Fill',
                line=dict(color='#FF8F00'),
                marker=dict(size=8)
            )
        )
        
        # Resaltar puntos imputados
        for idx in [3, 7]:
            fig.add_trace(
                go.Scatter(
                    x=[df_demo['fecha'][idx]],
                    y=[df_interp_linear['valor'][idx]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='#26A69A',
                        line=dict(width=2, color='black')
                    ),
                    name='Valor Imputado (Lineal)' if idx == 3 else None,
                    showlegend=idx == 3
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[df_demo['fecha'][idx]],
                    y=[df_interp_ffill['valor'][idx]],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='#FF8F00',
                        line=dict(width=2, color='black')
                    ),
                    name='Valor Imputado (FFill)' if idx == 3 else None,
                    showlegend=idx == 3
                )
            )
        
        fig.update_layout(
            title="Comparación de Métodos de Imputación",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Detección de Outliers")
        
        st.markdown("""
        Los outliers (valores atípicos) pueden afectar significativamente el modelado de series temporales.
        Es importante identificarlos y decidir cómo tratarlos.
        """)
        
        # Métodos de detección de outliers
        st.markdown("#### Métodos de Detección de Outliers")
        
        metodo_outlier = st.radio(
            "Seleccione método de detección:",
            ["Rango Intercuartílico (IQR)", "Z-Score", "Desviación Absoluta de la Mediana (MAD)"]
        )
        
        # Explicación del método seleccionado
        if metodo_outlier == "Rango Intercuartílico (IQR)":
            st.markdown("""
            **Método IQR**:
            1. Calcular Q1 (primer cuartil) y Q3 (tercer cuartil)
            2. Calcular IQR = Q3 - Q1
            3. Definir límites: Límite inferior = Q1 - 1.5*IQR, Límite superior = Q3 + 1.5*IQR
            4. Identificar outliers: valores < Límite inferior o > Límite superior
            """)
            
            # Código para IQR
            with st.expander("Ver código para detección con IQR"):
                st.code("""
# Calcular cuartiles e IQR
Q1 = df['IVA_TOTAL'].quantile(0.25)
Q3 = df['IVA_TOTAL'].quantile(0.75)
IQR = Q3 - Q1

# Definir límites
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identificar outliers
df['outlier'] = ((df['IVA_TOTAL'] < lower_bound) | (df['IVA_TOTAL'] > upper_bound))
                """)
        
        elif metodo_outlier == "Z-Score":
            st.markdown("""
            **Método Z-Score**:
            1. Calcular la media y desviación estándar de la serie
            2. Calcular el Z-Score para cada punto: Z = (X - media) / desviación estándar
            3. Definir un umbral (típicamente 3)
            4. Identificar outliers: valores con |Z| > umbral
            """)
            
            # Código para Z-Score
            with st.expander("Ver código para detección con Z-Score"):
                st.code("""
# Calcular media y desviación estándar
mean = df['IVA_TOTAL'].mean()
std = df['IVA_TOTAL'].std()

# Calcular Z-Score
df['zscore'] = (df['IVA_TOTAL'] - mean) / std

# Identificar outliers
threshold = 3
df['outlier'] = abs(df['zscore']) > threshold
                """)
        
        else:  # MAD
            st.markdown("""
            **Método MAD (Desviación Absoluta de la Mediana)**:
            1. Calcular la mediana de la serie
            2. Calcular las desviaciones absolutas respecto a la mediana
            3. Calcular la mediana de estas desviaciones (MAD)
            4. Calcular el score modificado: 0.6745 * (X - mediana) / MAD
            5. Identificar outliers: valores con |score| > umbral (típicamente 3.5)
            """)
            
            # Código para MAD
            with st.expander("Ver código para detección con MAD"):
                st.code("""
# Calcular mediana
median = df['IVA_TOTAL'].median()

# Calcular desviaciones absolutas
abs_dev = abs(df['IVA_TOTAL'] - median)

# Calcular MAD
mad = abs_dev.median()

# Calcular score modificado
modified_zscore = 0.6745 * (df['IVA_TOTAL'] - median) / mad

# Identificar outliers
threshold = 3.5
df['outlier'] = abs(modified_zscore) > threshold
                """)
        
        # Visualización de outliers en los datos
        st.markdown("#### Visualización de Outliers en IVA Total")
        
        # Usar los datos procesados que ya tienen la columna de outliers
        if 'IVA_outlier' in df_procesado.columns:
            # Preparar datos para visualización
            if 'fecha' not in df_procesado.columns:
                df_viz = df_procesado.copy()
                
                # Convertir trimestres a fechas si es necesario
                if 'TRIMESTRE' in df_procesado.columns:
                    def trimestre_a_fecha(trimestre):
                        año, q = trimestre.split('-')
                        # Asegurar que q es un número válido
                        q_num = int(q) if q.isdigit() else int(q[-1])
                        mes = (q_num - 1) * 3 + 1
                        return f"{año}-{mes:02d}-01"
                    
                    df_viz['fecha'] = df_viz['TRIMESTRE'].apply(trimestre_a_fecha)
                    df_viz['fecha'] = pd.to_datetime(df_viz['fecha'])
                else:
                    df_viz['fecha'] = pd.to_datetime(df_viz.index)
            else:
                df_viz = df_procesado.copy()
                df_viz['fecha'] = pd.to_datetime(df_viz['fecha'])
            
            # Crear gráfico de outliers
            fig = go.Figure()
            
            # Añadir línea para todos los puntos
            fig.add_trace(
                go.Scatter(
                    x=df_viz['fecha'],
                    y=df_viz['IVA_TOTAL'],
                    mode='lines+markers',
                    name='IVA Total',
                    line=dict(color='#1E88E5'),
                    marker=dict(
                        size=8,
                        color=np.where(df_viz['IVA_outlier'], 'red', '#1E88E5'),
                        line=dict(width=2, color=np.where(df_viz['IVA_outlier'], 'black', 'rgba(0,0,0,0)'))
                    )
                )
            )
            
            # Añadir puntos específicos para outliers
            outliers = df_viz[df_viz['IVA_outlier']]
            if not outliers.empty:
                fig.add_trace(
                    go.Scatter(
                        x=outliers['fecha'],
                        y=outliers['IVA_TOTAL'],
                        mode='markers',
                        name='Outliers',
                        marker=dict(
                            size=12,
                            color='red',
                            symbol='circle-open',
                            line=dict(width=2, color='black')
                        )
                    )
                )
            
            # Añadir línea vertical para COVID si está en el rango de fechas
            covid_date = pd.to_datetime('2020-01-01')
            # Asegurar que las fechas sean datetime para comparación
            df_viz_dates = pd.to_datetime(df_viz['fecha'])
            if (df_viz_dates.min() <= covid_date) and (df_viz_dates.max() >= covid_date):
                # Usar shapes en lugar de add_vline para evitar problemas de tipos
                fig.add_shape(
                    type="line",
                    x0=covid_date,
                    x1=covid_date,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(dash="dash", color="gray", width=2)
                )
                # Añadir anotación manualmente
                fig.add_annotation(
                    x=covid_date,
                    y=1,
                    yref="paper",
                    text="Inicio COVID-19",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="gray",
                    ax=20,
                    ay=-30
                )
            
            fig.update_layout(
                title="Detección de Outliers en IVA Total",
                xaxis_title="Fecha",
                yaxis_title="IVA Total",
                height=500,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar información sobre los outliers detectados
            if outliers.empty:
                st.success("No se detectaron outliers en la serie de IVA Total.")
            else:
                st.warning(f"Se detectaron {len(outliers)} outliers en la serie de IVA Total.")
                st.dataframe(
                    outliers[['fecha', 'TRIMESTRE', 'IVA_TOTAL']].sort_values('fecha'),
                    use_container_width=True
                )
        else:
            st.info("Los datos procesados no contienen la columna de outliers. Utilice el método de preprocesamiento para detectarlos.")
    
    with tab4:
        st.markdown("### Transformaciones para Series Temporales")
        
        st.markdown("""
        Las transformaciones son fundamentales para preparar series temporales para el modelado.
        Algunas transformaciones comunes incluyen:
        """)
        
        # Selección de transformación
        transformacion = st.selectbox(
            "Seleccione transformación:",
            ["Cambio Porcentual", "Diferenciación", "Logaritmo", "Normalización"]
        )
        
        # Explicación y visualización según la transformación seleccionada
        if transformacion == "Cambio Porcentual":
            st.markdown("""
            **Cambio Porcentual**:
            
            Calcula la variación porcentual respecto al valor anterior. Es útil para:
            - Eliminar tendencias y enfocarse en tasas de cambio
            - Hacer comparables series de diferentes magnitudes
            - Identificar patrones de crecimiento/decrecimiento
            """)
            
            # Código para cambio porcentual
            with st.expander("Ver código para cambio porcentual"):
                st.code("""
# Calcular cambio porcentual
df['IVA_pct_change'] = df['IVA_TOTAL'].pct_change() * 100
df['PIB_pct_change'] = df['PIB'].pct_change() * 100
                """)
            
            # Visualización de cambio porcentual
            if 'IVA_pct_change' in df_procesado.columns and 'PIB_pct_change' in df_procesado.columns:
                # Preparar datos para visualización
                if 'fecha' not in df_procesado.columns:
                    df_viz = df_procesado.copy()
                    
                    # Convertir trimestres a fechas si es necesario
                    if 'TRIMESTRE' in df_procesado.columns:
                        def trimestre_a_fecha(trimestre):
                            año, q = trimestre.split('-')
                            # Asegurar que q es un número válido
                            q_num = int(q) if q.isdigit() else int(q[-1])
                            mes = (q_num - 1) * 3 + 1
                            return f"{año}-{mes:02d}-01"
                        
                        df_viz['fecha'] = df_viz['TRIMESTRE'].apply(trimestre_a_fecha)
                        df_viz['fecha'] = pd.to_datetime(df_viz['fecha'])
                    else:
                        df_viz['fecha'] = pd.to_datetime(df_viz.index)
                else:
                    df_viz = df_procesado.copy()
                    df_viz['fecha'] = pd.to_datetime(df_viz['fecha'])
                
                # Crear gráfico de cambio porcentual
                fig = make_subplots(
                    rows=2, 
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=("Cambio Porcentual en IVA Total", "Cambio Porcentual en PIB")
                )
                
                # Añadir línea para IVA
                fig.add_trace(
                    go.Scatter(
                        x=df_viz['fecha'],
                        y=df_viz['IVA_pct_change'],
                        mode='lines+markers',
                        name='IVA % Cambio',
                        line=dict(color='#1E88E5'),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
                
                # Añadir línea para PIB
                fig.add_trace(
                    go.Scatter(
                        x=df_viz['fecha'],
                        y=df_viz['PIB_pct_change'],
                        mode='lines+markers',
                        name='PIB % Cambio',
                        line=dict(color='#26A69A'),
                        marker=dict(size=6)
                    ),
                    row=2, col=1
                )
                
                # Añadir línea horizontal en cero
                fig.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="gray",
                    row=1, col=1
                )
                
                fig.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="gray",
                    row=2, col=1
                )
                
                # Añadir línea vertical para COVID si está en el rango de fechas
                covid_date = pd.to_datetime('2020-01-01')
                # Asegurar que las fechas sean datetime para comparación
                df_viz_dates = pd.to_datetime(df_viz['fecha'])
                if (df_viz_dates.min() <= covid_date) and (df_viz_dates.max() >= covid_date):
                    # Usar shapes en lugar de add_vline para evitar problemas de tipos
                    fig.add_shape(
                        type="line",
                        x0=covid_date,
                        x1=covid_date,
                        y0=0,
                        y1=1,
                        yref="y",
                        line=dict(dash="dash", color="gray", width=2),
                        row=1, col=1
                    )
                    fig.add_annotation(
                        x=covid_date,
                        y=1,
                        yref="y domain",
                        text="Inicio COVID-19",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="gray",
                        ax=20,
                        ay=-30,
                        row=1, col=1
                    )
                    
                    fig.add_shape(
                        type="line",
                        x0=covid_date,
                        x1=covid_date,
                        y0=0,
                        y1=1,
                        yref="y2",
                        line=dict(dash="dash", color="gray", width=2),
                        row=2, col=1
                    )
                
                fig.update_layout(
                    height=600,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                fig.update_xaxes(title_text="Fecha", row=2, col=1)
                fig.update_yaxes(title_text="Cambio %", row=1, col=1)
                fig.update_yaxes(title_text="Cambio %", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Los datos procesados no contienen las columnas de cambio porcentual.")
        
        elif transformacion == "Diferenciación":
            st.markdown("""
            **Diferenciación**:
            
            Calcula la diferencia entre valores consecutivos. Es útil para:
            - Eliminar tendencias y hacer la serie estacionaria
            - Preparar datos para modelos como ARIMA
            - Identificar cambios absolutos en lugar de relativos
            """)
            
            # Código para diferenciación
            with st.expander("Ver código para diferenciación"):
                st.code("""
# Calcular primera diferencia
df['IVA_diff'] = df['IVA_TOTAL'].diff()
df['PIB_diff'] = df['PIB'].diff()

# Calcular segunda diferencia si es necesario
df['IVA_diff2'] = df['IVA_TOTAL'].diff().diff()
                """)
            
            # Demostración de diferenciación
            # Usar los datos originales para mostrar el efecto
            df_demo = df_original.copy()
            
            # Convertir trimestres a fechas
            def trimestre_a_fecha(trimestre):
                año, q = trimestre.split('-')
                # Asegurar que q es un número válido
                q_num = int(q) if q.isdigit() else int(q[-1])
                mes = (q_num - 1) * 3 + 1
                return f"{año}-{mes:02d}-01"
            
            df_demo['fecha'] = df_demo['TRIMESTRE'].apply(trimestre_a_fecha)
            df_demo['fecha'] = pd.to_datetime(df_demo['fecha'])
            
            # Calcular diferencias
            df_demo['IVA_diff'] = df_demo['IVA_TOTAL'].diff()
            df_demo['IVA_diff2'] = df_demo['IVA_TOTAL'].diff().diff()
            
            # Seleccionar orden de diferenciación
            orden_diff = st.radio(
                "Orden de diferenciación:",
                ["Primera diferencia", "Segunda diferencia"]
            )
            
            # Crear gráfico comparativo
            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(
                    "Serie Original (IVA Total)",
                    "Primera diferencia" if orden_diff == "Primera diferencia" else "Segunda diferencia"
                )
            )
            
            # Añadir serie original
            fig.add_trace(
                go.Scatter(
                    x=df_demo['fecha'],
                    y=df_demo['IVA_TOTAL'],
                    mode='lines+markers',
                    name='IVA Total',
                    line=dict(color='#1E88E5'),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            # Añadir serie diferenciada
            if orden_diff == "Primera diferencia":
                fig.add_trace(
                    go.Scatter(
                        x=df_demo['fecha'],
                        y=df_demo['IVA_diff'],
                        mode='lines+markers',
                        name='Primera diferencia',
                        line=dict(color='#26A69A'),
                        marker=dict(size=6)
                    ),
                    row=2, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df_demo['fecha'],
                        y=df_demo['IVA_diff2'],
                        mode='lines+markers',
                        name='Segunda diferencia',
                        line=dict(color='#FF8F00'),
                        marker=dict(size=6)
                    ),
                    row=2, col=1
                )
            
            # Añadir línea horizontal en cero para la diferenciación
            fig.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="gray",
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            fig.update_xaxes(title_text="Fecha", row=2, col=1)
            fig.update_yaxes(title_text="IVA Total", row=1, col=1)
            fig.update_yaxes(title_text="Diferencia", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Observaciones**:
            - La primera diferencia elimina tendencias lineales
            - La segunda diferencia elimina tendencias cuadráticas
            - Una serie diferenciada que fluctúa alrededor de cero indica estacionariedad
            - La diferenciación excesiva puede introducir autocorrelación negativa
            """)
        
        elif transformacion == "Logaritmo":
            st.markdown("""
            **Transformación Logarítmica**:
            
            Aplica el logaritmo natural a los valores. Es útil para:
            - Estabilizar la varianza en series con crecimiento exponencial
            - Hacer que distribuciones asimétricas sean más simétricas
            - Reducir el impacto de valores extremos
            - Convertir relaciones multiplicativas en aditivas
            """)
            
            # Código para transformación logarítmica
            with st.expander("Ver código para transformación logarítmica"):
                st.code("""
# Aplicar transformación logarítmica
import numpy as np
df['IVA_log'] = np.log(df['IVA_TOTAL'])
df['PIB_log'] = np.log(df['PIB'])
                """)
            
            # Demostración de transformación logarítmica
            df_demo = df_original.copy()
            
            # Convertir trimestres a fechas
            def trimestre_a_fecha(trimestre):
                año, q = trimestre.split('-')
                # Asegurar que q es un número válido
                q_num = int(q) if q.isdigit() else int(q[-1])
                mes = (q_num - 1) * 3 + 1
                return f"{año}-{mes:02d}-01"
            
            df_demo['fecha'] = df_demo['TRIMESTRE'].apply(trimestre_a_fecha)
            df_demo['fecha'] = pd.to_datetime(df_demo['fecha'])
            
            # Calcular transformación logarítmica
            df_demo['IVA_log'] = np.log(df_demo['IVA_TOTAL'])
            df_demo['PIB_log'] = np.log(df_demo['PIB'])
            
            # Seleccionar variable
            variable_log = st.selectbox(
                "Seleccione variable:",
                ["IVA Total", "PIB"]
            )
            
            # Crear gráfico comparativo
            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(
                    f"Serie Original ({variable_log})",
                    f"Transformación Logarítmica ({variable_log})"
                )
            )
            
            # Añadir serie original y transformada según selección
            if variable_log == "IVA Total":
                fig.add_trace(
                    go.Scatter(
                        x=df_demo['fecha'],
                        y=df_demo['IVA_TOTAL'],
                        mode='lines+markers',
                        name='IVA Total',
                        line=dict(color='#1E88E5'),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_demo['fecha'],
                        y=df_demo['IVA_log'],
                        mode='lines+markers',
                        name='Log(IVA Total)',
                        line=dict(color='#26A69A'),
                        marker=dict(size=6)
                    ),
                    row=2, col=1
                )
            else:  # PIB
                fig.add_trace(
                    go.Scatter(
                        x=df_demo['fecha'],
                        y=df_demo['PIB'],
                        mode='lines+markers',
                        name='PIB',
                        line=dict(color='#FF8F00'),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_demo['fecha'],
                        y=df_demo['PIB_log'],
                        mode='lines+markers',
                        name='Log(PIB)',
                        line=dict(color='#26A69A'),
                        marker=dict(size=6)
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            fig.update_xaxes(title_text="Fecha", row=2, col=1)
            fig.update_yaxes(title_text=variable_log, row=1, col=1)
            fig.update_yaxes(title_text=f"Log({variable_log})", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Observaciones**:
            - La transformación logarítmica comprime valores altos y expande valores bajos
            - Útil cuando la serie muestra crecimiento exponencial
            - Facilita la visualización de tasas de crecimiento relativas
            - Al aplicar diferencias a datos logarítmicos, se obtienen aproximaciones de tasas de crecimiento
            """)
        
        else:  # Normalización
            st.markdown("""
            **Normalización**:
            
            Escala los datos a un rango específico, típicamente [0,1] o [-1,1]. Es útil para:
            - Preparar datos para modelos de machine learning, especialmente redes neuronales
            - Hacer comparables variables con diferentes escalas
            - Mejorar la convergencia de algoritmos de optimización
            """)
            
            # Código para normalización
            with st.expander("Ver código para normalización"):
                st.code("""
# Normalización Min-Max (escala [0,1])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['IVA_norm', 'PIB_norm']] = scaler.fit_transform(df[['IVA_TOTAL', 'PIB']])

# Normalización Z-Score (media 0, desviación estándar 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['IVA_std', 'PIB_std']] = scaler.fit_transform(df[['IVA_TOTAL', 'PIB']])
                """)
            
            # Demostración de normalización
            df_demo = df_original.copy()
            
            # Convertir trimestres a fechas
            def trimestre_a_fecha(trimestre):
                año, q = trimestre.split('-')
                # Asegurar que q es un número válido
                q_num = int(q) if q.isdigit() else int(q[-1])
                mes = (q_num - 1) * 3 + 1
                return f"{año}-{mes:02d}-01"
            
            df_demo['fecha'] = df_demo['TRIMESTRE'].apply(trimestre_a_fecha)
            df_demo['fecha'] = pd.to_datetime(df_demo['fecha'])
            
            # Calcular normalizaciones
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
            
            # Min-Max
            min_max_scaler = MinMaxScaler()
            df_demo[['IVA_norm', 'PIB_norm']] = min_max_scaler.fit_transform(df_demo[['IVA_TOTAL', 'PIB']])
            
            # Z-Score
            std_scaler = StandardScaler()
            df_demo[['IVA_std', 'PIB_std']] = std_scaler.fit_transform(df_demo[['IVA_TOTAL', 'PIB']])
            
            # Seleccionar tipo de normalización
            tipo_norm = st.radio(
                "Tipo de normalización:",
                ["Min-Max (escala [0,1])", "Z-Score (media 0, std 1)"]
            )
            
            # Crear gráfico comparativo
            fig = go.Figure()
            
            # Añadir series según selección
            if tipo_norm == "Min-Max (escala [0,1])":
                fig.add_trace(
                    go.Scatter(
                        x=df_demo['fecha'],
                        y=df_demo['IVA_norm'],
                        mode='lines+markers',
                        name='IVA (Normalizado)',
                        line=dict(color='#1E88E5'),
                        marker=dict(size=6)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_demo['fecha'],
                        y=df_demo['PIB_norm'],
                        mode='lines+markers',
                        name='PIB (Normalizado)',
                        line=dict(color='#FF8F00'),
                        marker=dict(size=6)
                    )
                )
                
                fig.update_yaxes(title_text="Valor Normalizado [0,1]")
            else:  # Z-Score
                fig.add_trace(
                    go.Scatter(
                        x=df_demo['fecha'],
                        y=df_demo['IVA_std'],
                        mode='lines+markers',
                        name='IVA (Z-Score)',
                        line=dict(color='#1E88E5'),
                        marker=dict(size=6)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=df_demo['fecha'],
                        y=df_demo['PIB_std'],
                        mode='lines+markers',
                        name='PIB (Z-Score)',
                        line=dict(color='#FF8F00'),
                        marker=dict(size=6)
                    )
                )
                
                fig.update_yaxes(title_text="Z-Score")
            
            # Añadir línea horizontal en cero para Z-Score
            if tipo_norm == "Z-Score (media 0, std 1)":
                fig.add_hline(
                    y=0, 
                    line_dash="dash", 
                    line_color="gray"
                )
            
            fig.update_layout(
                title=f"Normalización {tipo_norm}",
                xaxis_title="Fecha",
                height=500,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar estadísticas de las series normalizadas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Estadísticas de IVA Normalizado**")
                if tipo_norm == "Min-Max (escala [0,1])":
                    st.dataframe(df_demo['IVA_norm'].describe().to_frame(), use_container_width=True)
                else:
                    st.dataframe(df_demo['IVA_std'].describe().to_frame(), use_container_width=True)
            
            with col2:
                st.markdown("**Estadísticas de PIB Normalizado**")
                if tipo_norm == "Min-Max (escala [0,1])":
                    st.dataframe(df_demo['PIB_norm'].describe().to_frame(), use_container_width=True)
                else:
                    st.dataframe(df_demo['PIB_std'].describe().to_frame(), use_container_width=True)
            
            st.markdown("""
            **Observaciones**:
            - La normalización Min-Max escala los datos al rango [0,1], preservando la forma de la distribución
            - La normalización Z-Score centra los datos en 0 con desviación estándar 1
            - Z-Score es más robusta frente a outliers que Min-Max
            - Para redes LSTM, la normalización es crucial para evitar problemas de convergencia
            """)
        
        # Información sobre la importancia de las transformaciones para LSTM
        st.markdown("---")
        st.markdown("### Importancia de las Transformaciones para Modelos LSTM")
        
        st.markdown("""
        Los modelos LSTM son sensibles a la escala de los datos de entrada. Las transformaciones adecuadas pueden:
        
        1. **Mejorar la convergencia**: Datos normalizados facilitan el entrenamiento de la red
        2. **Reducir el sesgo**: Transformaciones como el logaritmo pueden reducir el impacto de valores extremos
        3. **Capturar patrones relevantes**: La diferenciación o el cambio porcentual pueden revelar patrones que no son evidentes en los datos originales
        4. **Estabilizar la varianza**: Fundamental para series con variabilidad creciente
        
        Para el modelado LSTM de series temporales económicas, se recomienda:
        - Normalizar los datos (típicamente con Min-Max Scaler)
        - Considerar transformaciones logarítmicas para variables con crecimiento exponencial
        - Evaluar el uso de cambios porcentuales para capturar tasas de crecimiento
        """)
