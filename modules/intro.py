import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def show():
    """
    Muestra la página de introducción del tablero Streamlit.
    """
    st.markdown("## Introducción a Series Temporales con LSTM")
    
    # Contenido principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### ¿Qué son las Series Temporales?
        
        Las series temporales son secuencias de datos recopilados a intervalos regulares de tiempo. 
        El análisis de series temporales permite identificar patrones, tendencias y comportamientos 
        cíclicos en los datos, lo que facilita la predicción de valores futuros.
        
        ### Componentes de una Serie Temporal
        
        - **Tendencia**: Movimiento a largo plazo de la serie
        - **Estacionalidad**: Patrones que se repiten en intervalos fijos
        - **Ciclo**: Fluctuaciones no periódicas
        - **Irregularidad**: Variaciones aleatorias o impredecibles
        
        ### Redes LSTM para Series Temporales
        
        Las redes Long Short-Term Memory (LSTM) son un tipo de red neuronal recurrente 
        especialmente diseñada para el análisis de secuencias y series temporales. 
        A diferencia de las redes neuronales tradicionales, las LSTM pueden "recordar" 
        información durante largos períodos, lo que las hace ideales para capturar 
        dependencias temporales en los datos.
        """)
        
        st.markdown("""
        ### Caso de Estudio: IVA y PIB
        
        En este tablero, analizaremos series temporales trimestrales de:
        
        - **IVA Total**: Recaudación del Impuesto al Valor Agregado
        - **PIB**: Producto Interno Bruto
        
        Exploraremos la relación entre estas variables económicas, su comportamiento 
        a lo largo del tiempo y cómo se vieron afectadas por eventos como la pandemia 
        de COVID-19. Finalmente, implementaremos un modelo LSTM para predecir valores 
        futuros.
        """)
    
    with col2:
        st.image("https://miro.medium.com/max/1400/1*7B0iyRUxRzUSpymSn8sU1A.png", 
                 caption="Arquitectura de una red LSTM")
        
        st.markdown("### Objetivos del Análisis")
        st.markdown("""
        - Explorar y visualizar series temporales económicas
        - Identificar patrones, tendencias y anomalías
        - Analizar el impacto de eventos externos (COVID-19)
        - Implementar y evaluar modelos LSTM para predicción
        - Interpretar resultados y extraer conclusiones
        """)
    
    # Sección de navegación rápida
    st.markdown("---")
    st.markdown("### Navegación Rápida")
    
    st.info("💡 **Tip**: Utiliza el menú lateral para navegar entre las diferentes secciones del análisis.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Exploración de Datos**")
        st.write("Analiza los datos originales y sus características estadísticas.")
    
    with col2:
        st.markdown("**🔍 Análisis Exploratorio**")
        st.write("Visualiza patrones, tendencias y relaciones en los datos.")
    
    with col3:
        st.markdown("**🧠 Modelado LSTM**")
        st.write("Implementa y evalúa modelos de redes neuronales LSTM.")
    
    # Información adicional
    with st.expander("Conceptos Clave de Series Temporales"):
        st.markdown("""
        #### Estacionariedad
        Una serie temporal es estacionaria cuando sus propiedades estadísticas (media, varianza, autocorrelación) 
        no cambian con el tiempo. La estacionariedad es importante para muchos modelos de series temporales.
        
        #### Autocorrelación
        Medida de la correlación entre los valores de la serie en diferentes puntos de tiempo. 
        Ayuda a identificar patrones y dependencias temporales.
        
        #### Ventana Temporal (Window Size)
        En el contexto de LSTM, determina cuántos pasos de tiempo anteriores se consideran para predecir 
        el siguiente valor. La elección adecuada de este parámetro es crucial para el rendimiento del modelo.
        """)
