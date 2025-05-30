import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show(df_original, df_procesado):
    """
    Muestra la página de exploración de datos del tablero Streamlit.
    
    Args:
        df_original: DataFrame con los datos originales
        df_procesado: DataFrame con los datos procesados
    """
    st.markdown("## Exploración de Datos")
    
    # Crear pestañas para organizar el contenido
    tab1, tab2, tab3 = st.tabs(["Datos Originales", "Estadísticas Descriptivas", "Visualización Inicial"])
    
    with tab1:
        st.markdown("### Datos Originales")
        st.markdown("A continuación se muestran los datos originales de IVA y PIB por trimestre:")
        
        # Mostrar los primeros registros con un slider para ajustar la cantidad
        num_rows = st.slider("Número de filas a mostrar:", min_value=5, max_value=len(df_original), value=10, step=5)
        st.dataframe(df_original.head(num_rows), use_container_width=True)
        
        # Información sobre la estructura de los datos
        st.markdown("### Estructura de los Datos")
        
        # Crear un DataFrame con la información de las columnas
        col_info = pd.DataFrame({
            'Columna': df_original.columns,
            'Tipo de Dato': df_original.dtypes.astype(str),
            'Valores No Nulos': df_original.count().values,
            'Valores Únicos': [df_original[col].nunique() for col in df_original.columns]
        })
        
        st.dataframe(col_info, use_container_width=True)
        
        # Opción para descargar los datos
        csv = df_original.to_csv(index=False)
        st.download_button(
            label="Descargar datos originales como CSV",
            data=csv,
            file_name="datos_iva_pib.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.markdown("### Estadísticas Descriptivas")
        st.markdown("Resumen estadístico de las variables numéricas:")
        
        # Mostrar estadísticas descriptivas
        st.dataframe(df_original.describe(), use_container_width=True)
        
        # Información adicional sobre las variables
        st.markdown("### Información de Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### IVA Total")
            st.markdown(f"- **Mínimo**: {df_original['IVA_TOTAL'].min():,.2f}")
            st.markdown(f"- **Máximo**: {df_original['IVA_TOTAL'].max():,.2f}")
            st.markdown(f"- **Media**: {df_original['IVA_TOTAL'].mean():,.2f}")
            st.markdown(f"- **Mediana**: {df_original['IVA_TOTAL'].median():,.2f}")
            
            # Histograma de IVA
            fig_iva = px.histogram(
                df_original, 
                x="IVA_TOTAL",
                nbins=20,
                title="Distribución de IVA Total",
                labels={"IVA_TOTAL": "IVA Total", "count": "Frecuencia"},
                color_discrete_sequence=["#1E88E5"]
            )
            st.plotly_chart(fig_iva, use_container_width=True)
        
        with col2:
            st.markdown("#### PIB")
            st.markdown(f"- **Mínimo**: {df_original['PIB'].min():,.2f}")
            st.markdown(f"- **Máximo**: {df_original['PIB'].max():,.2f}")
            st.markdown(f"- **Media**: {df_original['PIB'].mean():,.2f}")
            st.markdown(f"- **Mediana**: {df_original['PIB'].median():,.2f}")
            
            # Histograma de PIB
            fig_pib = px.histogram(
                df_original, 
                x="PIB",
                nbins=20,
                title="Distribución de PIB",
                labels={"PIB": "PIB", "count": "Frecuencia"},
                color_discrete_sequence=["#26A69A"]
            )
            st.plotly_chart(fig_pib, use_container_width=True)
    
    with tab3:
        st.markdown("### Visualización Inicial de Series Temporales")
        
        # Preparar datos para visualización
        # Convertir TRIMESTRE a datetime si no lo está ya
        if df_original['TRIMESTRE'].dtype == 'object':
            # Crear una columna de fecha para la visualización
            df_viz = df_original.copy()
            
            # Convertir trimestres a fechas (asumiendo formato YYYY-Q)
            def trimestre_a_fecha(trimestre):
                año, q = trimestre.split('-')
                mes = (int(q[-1]) - 1) * 3 + 1  # Q1->1, Q2->4, Q3->7, Q4->10
                return f"{año}-{mes:02d}-01"
            
            df_viz['fecha'] = df_viz['TRIMESTRE'].apply(trimestre_a_fecha)
            df_viz['fecha'] = pd.to_datetime(df_viz['fecha'])
            df_viz = df_viz.sort_values('fecha')
        else:
            df_viz = df_original.copy()
            df_viz['fecha'] = pd.to_datetime(df_viz['TRIMESTRE'])
            df_viz = df_viz.sort_values('fecha')
        
        # Crear gráfico de series temporales
        st.markdown("Seleccione las variables a visualizar:")
        
        # Checkboxes para seleccionar variables
        col1, col2 = st.columns(2)
        with col1:
            show_iva = st.checkbox("IVA Total", value=True)
        with col2:
            show_pib = st.checkbox("PIB", value=True)
        
        # Crear figura con subplots
        fig = make_subplots(
            rows=2 if show_iva and show_pib else 1, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=([
                "Recaudación de IVA Total (Trimestral)" if show_iva else None,
                "PIB Trimestral" if show_pib else None
            ])
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
                row=2 if show_iva else 1, col=1
            )
        
        # Actualizar diseño
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode="x unified"
        )
        
        fig.update_xaxes(
            title_text="Fecha",
            tickformat="%Y-%m",
            tickangle=45,
            tickmode="auto",
            nticks=20
        )
        
        if show_iva:
            fig.update_yaxes(title_text="IVA Total", row=1, col=1)
        if show_pib:
            fig.update_yaxes(title_text="PIB", row=2 if show_iva else 1, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Añadir información sobre la interpretación
        with st.expander("Interpretación de las Series Temporales"):
            st.markdown("""
            ### Interpretación de las Series Temporales
            
            - **Tendencia**: Observe si hay un patrón general ascendente o descendente a lo largo del tiempo.
            - **Estacionalidad**: Busque patrones que se repiten en intervalos regulares (por ejemplo, cada año).
            - **Outliers**: Identifique valores atípicos que se desvían significativamente del patrón general.
            - **Cambios Estructurales**: Detecte cambios abruptos en el comportamiento de la serie (como el impacto del COVID-19).
            
            La visualización de series temporales es el primer paso para entender el comportamiento de los datos
            y determinar qué técnicas de modelado serán más apropiadas.
            """)
