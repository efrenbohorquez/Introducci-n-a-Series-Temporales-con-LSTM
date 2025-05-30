import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Agregar directorio actual al path para importaciones
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar módulos
from modules import intro, exploracion, preprocesamiento, analisis, modelado, resultados

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Series Temporales con LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar datos
@st.cache_data
def cargar_datos():
    """Carga los datos de los archivos CSV."""
    try:
        # Cargar datos procesados
        df_procesado = pd.read_csv('data/datos_procesados.csv')
        
        # Cargar datos de IVA y PIB
        df_iva_pib = pd.read_csv('data/datos_iva_pib.csv')
        
        return df_procesado, df_iva_pib
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None, None

def main():
    """Función principal que ejecuta la aplicación Streamlit."""
    
    # Título principal
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/260px-Scikit_learn_logo_small.svg.png", width=100)
    st.sidebar.title("Análisis de Series Temporales")
    st.sidebar.markdown("### Maestría en Ciencia de Datos")
    
    # Menú de navegación
    menu = st.sidebar.radio(
        "Seleccione una sección:",
        ["Introducción", "Exploración de Datos", "Preprocesamiento", 
         "Análisis Exploratorio", "Modelado LSTM", "Resultados y Conclusiones"]
    )
    
    # Cargar datos
    df_procesado, df_iva_pib = cargar_datos()
    
    # Verificar si los datos se cargaron correctamente
    if df_procesado is None or df_iva_pib is None:
        # Si no se pudieron cargar los datos desde los archivos, usar los datos de ejemplo
        st.warning("Usando datos de ejemplo para demostración. Para usar datos reales, coloque los archivos CSV en la carpeta 'data'.")
        
        # Crear datos de ejemplo
        # Fechas trimestrales desde 2015 hasta 2023
        fechas = pd.date_range(start='2015-01-01', end='2023-12-31', freq='Q')
        
        # Crear DataFrame de ejemplo para datos procesados
        df_procesado = pd.DataFrame({
            'fecha': fechas,
            'TRIMESTRE': [f"{fecha.year}-Q{fecha.quarter}" for fecha in fechas],
            'IVA_TOTAL': np.random.normal(10000000, 1000000, len(fechas)) * (1 + 0.1 * np.arange(len(fechas)) / len(fechas)),
            'PIB': np.random.normal(120000000, 5000000, len(fechas)) * (1 + 0.15 * np.arange(len(fechas)) / len(fechas)),
        })
        
        # Añadir cambios porcentuales
        df_procesado['IVA_pct_change'] = df_procesado['IVA_TOTAL'].pct_change() * 100
        df_procesado['PIB_pct_change'] = df_procesado['PIB'].pct_change() * 100
        
        # Usar el mismo DataFrame para ambos
        df_iva_pib = df_procesado.copy()
    
    # Mostrar la sección seleccionada
    if menu == "Introducción":
        intro.show()
    
    elif menu == "Exploración de Datos":
        exploracion.show(df_iva_pib, df_procesado)
    
    elif menu == "Preprocesamiento":
        preprocesamiento.show(df_iva_pib, df_procesado)
    
    elif menu == "Análisis Exploratorio":
        analisis.show(df_procesado)
    
    elif menu == "Modelado LSTM":
        modelado.show(df_procesado)
    
    elif menu == "Resultados y Conclusiones":
        resultados.show(df_procesado)
    
    # Pie de página
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Desarrollado para:")
    st.sidebar.markdown("Maestría en Ciencia de Datos")
    st.sidebar.markdown("© 2025")

if __name__ == "__main__":
    main()
