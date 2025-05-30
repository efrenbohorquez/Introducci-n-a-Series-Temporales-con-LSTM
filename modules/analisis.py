import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

def show(df_procesado):
    """
    Muestra la página de análisis exploratorio del tablero Streamlit.
    
    Args:
        df_procesado: DataFrame con los datos procesados
    """
    st.markdown("## Análisis Exploratorio")
    
    # Crear pestañas para organizar el contenido
    tab1, tab2, tab3 = st.tabs(["Series Temporales", "Segmentación COVID", "Correlación"])
    
    # Preparar datos para visualización
    if 'fecha' not in df_procesado.columns:
        df_viz = df_procesado.copy()
        
        # Convertir trimestres a fechas si es necesario
        if 'TRIMESTRE' in df_procesado.columns:
            def trimestre_a_fecha(trimestre):
                año, q = trimestre.split('-')
                mes = (int(q[-1]) - 1) * 3 + 1
                return f"{año}-{mes:02d}-01"
            
            df_viz['fecha'] = df_viz['TRIMESTRE'].apply(trimestre_a_fecha)
            df_viz['fecha'] = pd.to_datetime(df_viz['fecha'])
        else:
            df_viz['fecha'] = pd.to_datetime(df_viz.index)
    else:
        df_viz = df_procesado.copy()
        df_viz['fecha'] = pd.to_datetime(df_viz['fecha'])
    
    # Definir fecha de inicio de COVID
    covid_date = pd.to_datetime('2020-01-01')
    
    # Segmentar datos pre y post COVID
    df_pre_covid = df_viz[df_viz['fecha'] < covid_date].copy()
    df_post_covid = df_viz[df_viz['fecha'] >= covid_date].copy()
    
    with tab1:
        st.markdown("### Visualización de Series Temporales")
        
        # Selección de variables
        st.markdown("Seleccione las variables a visualizar:")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_iva = st.checkbox("IVA Total", value=True)
        with col2:
            show_pib = st.checkbox("PIB", value=True)
        with col3:
            show_iva_pct = st.checkbox("IVA % Cambio", value=False)
        with col4:
            show_pib_pct = st.checkbox("PIB % Cambio", value=False)
        
        # Determinar número de filas para subplots
        num_rows = sum([show_iva, show_pib, show_iva_pct, show_pib_pct])
        if num_rows == 0:
            st.warning("Por favor, seleccione al menos una variable para visualizar.")
            num_rows = 1
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=num_rows, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=[]
        )
        
        # Añadir trazas según selección
        row = 1
        
        if show_iva:
            fig.add_trace(
                go.Scatter(
                    x=df_viz['fecha'], 
                    y=df_viz['IVA_TOTAL'],
                    mode='lines+markers',
                    name='IVA Total',
                    line=dict(color='#1E88E5', width=2),
                    marker=dict(size=6)
                ),
                row=row, col=1
            )
            
            # Añadir título al subplot
            fig.update_yaxes(title_text="IVA Total", row=row, col=1)
            row += 1
        
        if show_pib:
            fig.add_trace(
                go.Scatter(
                    x=df_viz['fecha'], 
                    y=df_viz['PIB'],
                    mode='lines+markers',
                    name='PIB',
                    line=dict(color='#26A69A', width=2),
                    marker=dict(size=6)
                ),
                row=row, col=1
            )
            
            # Añadir título al subplot
            fig.update_yaxes(title_text="PIB", row=row, col=1)
            row += 1
        
        if show_iva_pct and 'IVA_pct_change' in df_viz.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_viz['fecha'], 
                    y=df_viz['IVA_pct_change'],
                    mode='lines+markers',
                    name='IVA % Cambio',
                    line=dict(color='#FF8F00', width=2),
                    marker=dict(size=6)
                ),
                row=row, col=1
            )
            
            # Añadir línea horizontal en cero
            fig.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="gray",
                row=row, col=1
            )
            
            # Añadir título al subplot
            fig.update_yaxes(title_text="IVA % Cambio", row=row, col=1)
            row += 1
        
        if show_pib_pct and 'PIB_pct_change' in df_viz.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_viz['fecha'], 
                    y=df_viz['PIB_pct_change'],
                    mode='lines+markers',
                    name='PIB % Cambio',
                    line=dict(color='#9C27B0', width=2),
                    marker=dict(size=6)
                ),
                row=row, col=1
            )
            
            # Añadir línea horizontal en cero
            fig.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="gray",
                row=row, col=1
            )
            
            # Añadir título al subplot
            fig.update_yaxes(title_text="PIB % Cambio", row=row, col=1)
        
        # Añadir línea vertical para COVID
        if (df_viz['fecha'].min() <= covid_date) and (df_viz['fecha'].max() >= covid_date):
            for i in range(1, num_rows + 1):
                fig.add_shape(
                    type="line",
                    x0=covid_date,
                    x1=covid_date,
                    y0=0,
                    y1=1,
                    yref=f"y{i}" if i > 1 else "y",
                    line=dict(dash="dash", color="red", width=2),
                    row=i, col=1
                )
                if i == 1:  # Solo añadir anotación en la primera fila
                    fig.add_annotation(
                        x=covid_date,
                        y=1,
                        yref=f"y{i} domain" if i > 1 else "y domain",
                        text="Inicio COVID-19",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="red",
                        ax=20,
                        ay=-30,
                        row=i, col=1
                    )
        
        # Actualizar diseño
        fig.update_layout(
            height=250 * num_rows,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified"
        )
        
        fig.update_xaxes(
            title_text="Fecha",
            tickformat="%Y-%m",
            tickangle=45,
            tickmode="auto",
            nticks=20,
            row=num_rows, col=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Añadir información sobre la interpretación
        with st.expander("Interpretación de las Series Temporales"):
            st.markdown("""
            ### Interpretación de las Series Temporales
            
            - **Tendencia**: Observe el patrón general ascendente o descendente a lo largo del tiempo.
            - **Estacionalidad**: Identifique patrones que se repiten en intervalos regulares (por ejemplo, cada año).
            - **Volatilidad**: Compare la variabilidad de las diferentes series y períodos.
            - **Cambios Estructurales**: Note cómo el comportamiento de las series cambia después del inicio de la pandemia COVID-19.
            - **Relación entre Variables**: Observe si los movimientos en IVA y PIB están correlacionados.
            
            Las visualizaciones de series temporales permiten identificar patrones y anomalías que pueden ser relevantes para el modelado predictivo.
            """)
    
    with tab2:
        st.markdown("### Segmentación Pre/Post COVID")
        
        st.markdown("""
        La pandemia de COVID-19 representó un punto de inflexión para muchas series temporales económicas.
        A continuación, analizamos cómo se comportaron las variables antes y después del inicio de la pandemia.
        """)
        
        # Selección de variable para análisis
        variable = st.selectbox(
            "Seleccione variable para análisis:",
            ["IVA Total", "PIB", "Ambas"]
        )
        
        if variable == "IVA Total" or variable == "Ambas":
            st.markdown("#### Análisis de IVA Total")
            
            # Estadísticas descriptivas pre/post COVID para IVA
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Estadísticas Pre-COVID**")
                st.dataframe(df_pre_covid['IVA_TOTAL'].describe().to_frame(), use_container_width=True)
            
            with col2:
                st.markdown("**Estadísticas Post-COVID**")
                st.dataframe(df_post_covid['IVA_TOTAL'].describe().to_frame(), use_container_width=True)
            
            # Visualización de distribuciones
            fig = make_subplots(
                rows=1, 
                cols=2,
                subplot_titles=("Distribución IVA Total (Pre-COVID)", "Distribución IVA Total (Post-COVID)")
            )
            
            # Histograma pre-COVID
            fig.add_trace(
                go.Histogram(
                    x=df_pre_covid['IVA_TOTAL'],
                    nbinsx=15,
                    marker_color='#1E88E5',
                    name='Pre-COVID'
                ),
                row=1, col=1
            )
            
            # Histograma post-COVID
            fig.add_trace(
                go.Histogram(
                    x=df_post_covid['IVA_TOTAL'],
                    nbinsx=15,
                    marker_color='#FF8F00',
                    name='Post-COVID'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                barmode='overlay'
            )
            
            fig.update_xaxes(title_text="IVA Total", row=1, col=1)
            fig.update_xaxes(title_text="IVA Total", row=1, col=2)
            fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
            fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Serie temporal con segmentación
            fig = go.Figure()
            
            # Datos pre-COVID
            fig.add_trace(
                go.Scatter(
                    x=df_pre_covid['fecha'],
                    y=df_pre_covid['IVA_TOTAL'],
                    mode='lines+markers',
                    name='Pre-COVID',
                    line=dict(color='#1E88E5', width=2),
                    marker=dict(size=8)
                )
            )
            
            # Datos post-COVID
            fig.add_trace(
                go.Scatter(
                    x=df_post_covid['fecha'],
                    y=df_post_covid['IVA_TOTAL'],
                    mode='lines+markers',
                    name='Post-COVID',
                    line=dict(color='#FF8F00', width=2),
                    marker=dict(size=8)
                )
            )
            
            # Línea vertical para COVID
            fig.add_shape(
                type="line",
                x0=covid_date,
                x1=covid_date,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dash", color="red", width=2)
            )
            fig.add_annotation(
                x=covid_date,
                y=1,
                yref="paper",
                text="Inicio COVID-19",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                ax=20,
                ay=-30
            )
            
            fig.update_layout(
                title="IVA Total: Pre vs Post COVID",
                xaxis_title="Fecha",
                yaxis_title="IVA Total",
                height=500,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        if variable == "PIB" or variable == "Ambas":
            st.markdown("#### Análisis de PIB")
            
            # Estadísticas descriptivas pre/post COVID para PIB
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Estadísticas Pre-COVID**")
                st.dataframe(df_pre_covid['PIB'].describe().to_frame(), use_container_width=True)
            
            with col2:
                st.markdown("**Estadísticas Post-COVID**")
                st.dataframe(df_post_covid['PIB'].describe().to_frame(), use_container_width=True)
            
            # Visualización de distribuciones
            fig = make_subplots(
                rows=1, 
                cols=2,
                subplot_titles=("Distribución PIB (Pre-COVID)", "Distribución PIB (Post-COVID)")
            )
            
            # Histograma pre-COVID
            fig.add_trace(
                go.Histogram(
                    x=df_pre_covid['PIB'],
                    nbinsx=15,
                    marker_color='#26A69A',
                    name='Pre-COVID'
                ),
                row=1, col=1
            )
            
            # Histograma post-COVID
            fig.add_trace(
                go.Histogram(
                    x=df_post_covid['PIB'],
                    nbinsx=15,
                    marker_color='#FF8F00',
                    name='Post-COVID'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                barmode='overlay'
            )
            
            fig.update_xaxes(title_text="PIB", row=1, col=1)
            fig.update_xaxes(title_text="PIB", row=1, col=2)
            fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
            fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Serie temporal con segmentación
            fig = go.Figure()
            
            # Datos pre-COVID
            fig.add_trace(
                go.Scatter(
                    x=df_pre_covid['fecha'],
                    y=df_pre_covid['PIB'],
                    mode='lines+markers',
                    name='Pre-COVID',
                    line=dict(color='#26A69A', width=2),
                    marker=dict(size=8)
                )
            )
            
            # Datos post-COVID
            fig.add_trace(
                go.Scatter(
                    x=df_post_covid['fecha'],
                    y=df_post_covid['PIB'],
                    mode='lines+markers',
                    name='Post-COVID',
                    line=dict(color='#FF8F00', width=2),
                    marker=dict(size=8)
                )
            )
            
            # Línea vertical para COVID
            fig.add_shape(
                type="line",
                x0=covid_date,
                x1=covid_date,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dash", color="red", width=2)
            )
            fig.add_annotation(
                x=covid_date,
                y=1,
                yref="paper",
                text="Inicio COVID-19",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                ax=20,
                ay=-30
            )
            
            fig.update_layout(
                title="PIB: Pre vs Post COVID",
                xaxis_title="Fecha",
                yaxis_title="PIB",
                height=500,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de cambios porcentuales si están disponibles
        if 'IVA_pct_change' in df_viz.columns and 'PIB_pct_change' in df_viz.columns:
            st.markdown("#### Análisis de Cambios Porcentuales")
            
            # Selección de variable para análisis de cambio porcentual
            pct_var = st.radio(
                "Seleccione variable para análisis de cambio porcentual:",
                ["IVA % Cambio", "PIB % Cambio"]
            )
            
            var_col = 'IVA_pct_change' if pct_var == "IVA % Cambio" else 'PIB_pct_change'
            var_name = "IVA" if pct_var == "IVA % Cambio" else "PIB"
            var_color = '#1E88E5' if pct_var == "IVA % Cambio" else '#26A69A'
            
            # Estadísticas descriptivas pre/post COVID para cambio porcentual
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Estadísticas Pre-COVID ({var_name} % Cambio)**")
                st.dataframe(df_pre_covid[var_col].describe().to_frame(), use_container_width=True)
            
            with col2:
                st.markdown(f"**Estadísticas Post-COVID ({var_name} % Cambio)**")
                st.dataframe(df_post_covid[var_col].describe().to_frame(), use_container_width=True)
            
            # Visualización de distribuciones
            fig = make_subplots(
                rows=1, 
                cols=2,
                subplot_titles=(f"Distribución {var_name} % Cambio (Pre-COVID)", f"Distribución {var_name} % Cambio (Post-COVID)")
            )
            
            # Histograma pre-COVID
            fig.add_trace(
                go.Histogram(
                    x=df_pre_covid[var_col],
                    nbinsx=15,
                    marker_color=var_color,
                    name='Pre-COVID'
                ),
                row=1, col=1
            )
            
            # Histograma post-COVID
            fig.add_trace(
                go.Histogram(
                    x=df_post_covid[var_col],
                    nbinsx=15,
                    marker_color='#FF8F00',
                    name='Post-COVID'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                barmode='overlay'
            )
            
            fig.update_xaxes(title_text=f"{var_name} % Cambio", row=1, col=1)
            fig.update_xaxes(title_text=f"{var_name} % Cambio", row=1, col=2)
            fig.update_yaxes(title_text="Frecuencia", row=1, col=1)
            fig.update_yaxes(title_text="Frecuencia", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Serie temporal con segmentación
            fig = go.Figure()
            
            # Datos pre-COVID
            fig.add_trace(
                go.Scatter(
                    x=df_pre_covid['fecha'],
                    y=df_pre_covid[var_col],
                    mode='lines+markers',
                    name='Pre-COVID',
                    line=dict(color=var_color, width=2),
                    marker=dict(size=8)
                )
            )
            
            # Datos post-COVID
            fig.add_trace(
                go.Scatter(
                    x=df_post_covid['fecha'],
                    y=df_post_covid[var_col],
                    mode='lines+markers',
                    name='Post-COVID',
                    line=dict(color='#FF8F00', width=2),
                    marker=dict(size=8)
                )
            )
            
            # Línea horizontal en cero
            fig.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="gray"
            )
            
            # Línea vertical para COVID
            fig.add_shape(
                type="line",
                x0=covid_date,
                x1=covid_date,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dash", color="red", width=2)
            )
            fig.add_annotation(
                x=covid_date,
                y=1,
                yref="paper",
                text="Inicio COVID-19",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                ax=20,
                ay=-30
            )
            
            fig.update_layout(
                title=f"{var_name} % Cambio: Pre vs Post COVID",
                xaxis_title="Fecha",
                yaxis_title=f"{var_name} % Cambio",
                height=500,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Conclusiones sobre el impacto de COVID
        st.markdown("#### Conclusiones sobre el Impacto de COVID-19")
        
        st.markdown("""
        El análisis de segmentación pre/post COVID-19 revela cambios significativos en el comportamiento de las series temporales:
        
        1. **Cambios en Niveles**: Se observan diferencias en los valores medios y medianas de las variables.
        2. **Cambios en Volatilidad**: La dispersión y variabilidad de los datos puede haber cambiado.
        3. **Cambios en Patrones Estacionales**: Los patrones cíclicos pueden haberse alterado.
        4. **Implicaciones para el Modelado**: Estos cambios estructurales sugieren que los modelos deben considerar explícitamente el efecto de la pandemia.
        
        Para el modelado LSTM, estos cambios implican que podría ser necesario:
        - Entrenar modelos separados para períodos pre y post COVID
        - Incluir variables dummy para capturar el efecto COVID
        - Dar mayor peso a datos recientes en el entrenamiento
        """)
    
    with tab3:
        st.markdown("### Análisis de Correlación")
        
        st.markdown("""
        El análisis de correlación permite entender la relación entre variables y cómo estas interactúan a lo largo del tiempo.
        """)
        
        # Selección de período para análisis de correlación
        periodo_corr = st.radio(
            "Seleccione período para análisis de correlación:",
            ["Todo el período", "Pre-COVID", "Post-COVID"]
        )
        
        # Seleccionar datos según período
        if periodo_corr == "Pre-COVID":
            df_corr = df_pre_covid.copy()
            titulo = "Correlación Pre-COVID"
        elif periodo_corr == "Post-COVID":
            df_corr = df_post_covid.copy()
            titulo = "Correlación Post-COVID"
        else:
            df_corr = df_viz.copy()
            titulo = "Correlación - Todo el período"
        
        # Calcular correlaciones
        if len(df_corr) > 0:
            # Seleccionar columnas numéricas para correlación
            numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
            
            # Filtrar columnas relevantes
            relevant_cols = [col for col in numeric_cols if col in ['IVA_TOTAL', 'PIB', 'IVA_pct_change', 'PIB_pct_change']]
            
            if len(relevant_cols) > 1:
                # Calcular matriz de correlación
                corr_matrix = df_corr[relevant_cols].corr()
                
                # Visualizar matriz de correlación como heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title=f"Matriz de Correlación - {titulo}"
                )
                
                fig.update_layout(
                    height=500,
                    width=700
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar correlaciones específicas
                st.markdown("#### Correlaciones Específicas")
                
                # Correlación IVA-PIB
                if 'IVA_TOTAL' in relevant_cols and 'PIB' in relevant_cols:
                    corr_iva_pib = corr_matrix.loc['IVA_TOTAL', 'PIB']
                    st.markdown(f"**Correlación IVA Total - PIB**: {corr_iva_pib:.4f}")
                
                # Correlación cambios porcentuales
                if 'IVA_pct_change' in relevant_cols and 'PIB_pct_change' in relevant_cols:
                    corr_pct = corr_matrix.loc['IVA_pct_change', 'PIB_pct_change']
                    st.markdown(f"**Correlación IVA % Cambio - PIB % Cambio**: {corr_pct:.4f}")
                
                # Visualización de dispersión
                st.markdown("#### Gráfico de Dispersión")
                
                # Selección de variables para dispersión
                col1, col2 = st.columns(2)
                
                with col1:
                    x_var = st.selectbox(
                        "Variable X:",
                        options=relevant_cols,
                        index=relevant_cols.index('PIB') if 'PIB' in relevant_cols else 0
                    )
                
                with col2:
                    y_var = st.selectbox(
                        "Variable Y:",
                        options=relevant_cols,
                        index=relevant_cols.index('IVA_TOTAL') if 'IVA_TOTAL' in relevant_cols else 0
                    )
                
                # Crear gráfico de dispersión
                fig = px.scatter(
                    df_corr,
                    x=x_var,
                    y=y_var,
                    trendline="ols",
                    title=f"Relación entre {y_var} y {x_var} - {titulo}",
                    labels={x_var: x_var, y_var: y_var},
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Añadir información sobre la interpretación
                with st.expander("Interpretación de Correlaciones"):
                    st.markdown("""
                    ### Interpretación de Correlaciones
                    
                    - **Correlación Positiva Fuerte (cercana a 1)**: Indica que ambas variables tienden a aumentar o disminuir juntas.
                    - **Correlación Negativa Fuerte (cercana a -1)**: Indica que cuando una variable aumenta, la otra tiende a disminuir.
                    - **Correlación Débil (cercana a 0)**: Indica poca o ninguna relación lineal entre las variables.
                    
                    **Consideraciones Importantes**:
                    - La correlación mide asociación lineal, no causalidad.
                    - Correlaciones pueden cambiar en diferentes períodos o condiciones económicas.
                    - Para series temporales, es importante considerar posibles correlaciones con rezagos.
                    
                    **Implicaciones para el Modelado LSTM**:
                    - Variables altamente correlacionadas pueden proporcionar información redundante.
                    - Cambios en patrones de correlación (como pre/post COVID) pueden requerir diferentes enfoques de modelado.
                    - La correlación entre variables puede ayudar a seleccionar características relevantes para el modelo.
                    """)
            else:
                st.warning("No hay suficientes variables numéricas relevantes para calcular correlaciones.")
        else:
            st.warning(f"No hay datos suficientes para el período seleccionado ({periodo_corr}).")
        
        # Análisis de autocorrelación
        st.markdown("### Análisis de Autocorrelación")
        
        st.markdown("""
        La autocorrelación mide la correlación de una serie temporal con versiones rezagadas de sí misma.
        Es fundamental para entender la dependencia temporal en los datos.
        """)
        
        # Selección de variable para autocorrelación
        var_acf = st.selectbox(
            "Seleccione variable para análisis de autocorrelación:",
            ["IVA Total", "PIB", "IVA % Cambio", "PIB % Cambio"]
        )
        
        # Mapear selección a columna
        if var_acf == "IVA Total":
            col_acf = 'IVA_TOTAL'
            color_acf = '#1E88E5'
        elif var_acf == "PIB":
            col_acf = 'PIB'
            color_acf = '#26A69A'
        elif var_acf == "IVA % Cambio":
            col_acf = 'IVA_pct_change'
            color_acf = '#FF8F00'
        else:
            col_acf = 'PIB_pct_change'
            color_acf = '#9C27B0'
        
        # Verificar si la columna existe
        if col_acf in df_viz.columns:
            # Número de rezagos para autocorrelación
            max_lags = st.slider(
                "Número de rezagos:",
                min_value=1,
                max_value=20,
                value=12
            )
            
            # Calcular autocorrelación
            # Eliminar NaN para cálculo correcto
            serie = df_viz[col_acf].dropna()
            
            if len(serie) > max_lags:
                # Calcular autocorrelaciones
                acf_values = [1.0]  # Lag 0 siempre es 1
                for lag in range(1, max_lags + 1):
                    # Calcular autocorrelación manualmente
                    # Desplazar la serie
                    y1 = serie[lag:]
                    y2 = serie[:-lag] if lag > 0 else serie
                    
                    # Asegurar misma longitud
                    min_len = min(len(y1), len(y2))
                    y1 = y1[:min_len]
                    y2 = y2[:min_len]
                    
                    # Calcular correlación
                    acf = np.corrcoef(y1, y2)[0, 1] if min_len > 1 else np.nan
                    acf_values.append(acf)
                
                # Crear gráfico de autocorrelación
                fig = go.Figure()
                
                # Añadir barras para autocorrelación
                fig.add_trace(
                    go.Bar(
                        x=list(range(max_lags + 1)),
                        y=acf_values,
                        marker_color=color_acf,
                        name='Autocorrelación'
                    )
                )
                
                # Añadir líneas de significancia (aproximadamente ±2/√n)
                n = len(serie)
                sig_level = 2 / np.sqrt(n)
                
                fig.add_hline(
                    y=sig_level,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Nivel de significancia (95%)",
                    annotation_position="top right"
                )
                
                fig.add_hline(
                    y=-sig_level,
                    line_dash="dash",
                    line_color="red"
                )
                
                fig.update_layout(
                    title=f"Función de Autocorrelación (ACF) - {var_acf}",
                    xaxis_title="Rezago",
                    yaxis_title="Autocorrelación",
                    height=500,
                    yaxis=dict(range=[-1, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Añadir información sobre la interpretación
                with st.expander("Interpretación de Autocorrelación"):
                    st.markdown("""
                    ### Interpretación de Autocorrelación
                    
                    La función de autocorrelación (ACF) muestra la correlación entre una serie temporal y sus valores rezagados:
                    
                    - **Autocorrelación Alta en Rezagos Bajos**: Indica fuerte dependencia a corto plazo.
                    - **Patrón Sinusoidal**: Sugiere comportamiento estacional o cíclico.
                    - **Decaimiento Lento**: Indica posible no estacionariedad o memoria larga.
                    - **Picos en Rezagos Específicos**: Sugieren estacionalidad con ese período.
                    
                    **Implicaciones para el Modelado LSTM**:
                    - La estructura de autocorrelación ayuda a determinar el tamaño óptimo de ventana temporal.
                    - Autocorrelaciones significativas indican qué rezagos son importantes para la predicción.
                    - Patrones estacionales identificados pueden incorporarse en la arquitectura del modelo.
                    
                    **Nota**: Las líneas punteadas rojas representan aproximadamente el nivel de significancia del 95%.
                    Autocorrelaciones que exceden estas líneas se consideran estadísticamente significativas.
                    """)
            else:
                st.warning(f"No hay suficientes datos para calcular autocorrelación con {max_lags} rezagos.")
        else:
            st.warning(f"La variable seleccionada ({var_acf}) no está disponible en los datos.")
