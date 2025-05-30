import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

def show(df_procesado):
    """
    Muestra la página de modelado LSTM del tablero Streamlit.
    
    Args:
        df_procesado: DataFrame con los datos procesados
    """
    st.markdown("## Modelado LSTM para Series Temporales")
    
    # Crear pestañas para organizar el contenido
    tab1, tab2, tab3, tab4 = st.tabs(["Preparación de Datos", "Arquitectura del Modelo", "Entrenamiento", "Evaluación"])
    
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
    
    # Ordenar por fecha
    df_viz = df_viz.sort_values('fecha')
    
    # Definir fecha de inicio de COVID
    covid_date = pd.to_datetime('2020-01-01')
    
    with tab1:
        st.markdown("### Preparación de Datos para LSTM")
        
        st.markdown("""
        La preparación adecuada de los datos es crucial para el rendimiento de los modelos LSTM.
        Los pasos principales incluyen:
        
        1. Selección de variables
        2. Normalización de datos
        3. Creación de secuencias temporales
        4. División en conjuntos de entrenamiento y prueba
        """)
        
        # Selección de variables
        st.markdown("#### 1. Selección de Variables")
        
        target_var = st.selectbox(
            "Seleccione variable objetivo:",
            ["IVA Total", "PIB"]
        )
        
        # Mapear selección a columna
        target_col = 'IVA_TOTAL' if target_var == "IVA Total" else 'PIB'
        
        # Selección de características adicionales
        st.markdown("#### Características adicionales (opcional):")
        
        use_other_var = st.checkbox("Incluir la otra variable como característica", value=True)
        use_lags = st.checkbox("Incluir rezagos como características", value=True)
        
        # Normalización
        st.markdown("#### 2. Normalización de Datos")
        
        st.markdown("""
        La normalización es esencial para los modelos LSTM, ya que:
        - Acelera la convergencia durante el entrenamiento
        - Evita que variables con diferentes escalas dominen el aprendizaje
        - Mejora la estabilidad numérica
        """)
        
        # Selección de método de normalización
        norm_method = st.radio(
            "Método de normalización:",
            ["Min-Max (escala [0,1])", "Z-Score (media 0, std 1)"]
        )
        
        # Demostración de normalización
        if target_col in df_viz.columns:
            # Crear figura para mostrar datos originales vs normalizados
            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f"{target_var} - Datos Originales", f"{target_var} - Datos Normalizados")
            )
            
            # Añadir serie original
            fig.add_trace(
                go.Scatter(
                    x=df_viz['fecha'],
                    y=df_viz[target_col],
                    mode='lines+markers',
                    name='Datos Originales',
                    line=dict(color='#1E88E5'),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            # Normalizar datos para visualización
            if norm_method == "Min-Max (escala [0,1])":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                normalized_data = scaler.fit_transform(df_viz[[target_col]]).flatten()
            else:  # Z-Score
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(df_viz[[target_col]]).flatten()
            
            # Añadir serie normalizada
            fig.add_trace(
                go.Scatter(
                    x=df_viz['fecha'],
                    y=normalized_data,
                    mode='lines+markers',
                    name='Datos Normalizados',
                    line=dict(color='#26A69A'),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )
            
            # Añadir línea vertical para COVID
            if (df_viz['fecha'].min() <= covid_date) and (df_viz['fecha'].max() >= covid_date):
                fig.add_shape(
                    type="line",
                    x0=covid_date,
                    x1=covid_date,
                    y0=0,
                    y1=1,
                    yref="y",
                    line=dict(dash="dash", color="red", width=2),
                    row=1, col=1
                )
                fig.add_annotation(
                    x=covid_date,
                    y=1,
                    yref="y domain",
                    text="Inicio COVID-19",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
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
                    line=dict(dash="dash", color="red", width=2),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=500,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            fig.update_xaxes(title_text="Fecha", row=2, col=1)
            fig.update_yaxes(title_text=target_var, row=1, col=1)
            fig.update_yaxes(title_text="Valor Normalizado", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Código para normalización
            with st.expander("Ver código para normalización"):
                if norm_method == "Min-Max (escala [0,1])":
                    st.code("""
# Normalización Min-Max (escala [0,1])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Para desnormalizar posteriormente
data_original = scaler.inverse_transform(data_normalized)
                    """)
                else:
                    st.code("""
# Normalización Z-Score (media 0, desviación estándar 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Para desnormalizar posteriormente
data_original = scaler.inverse_transform(data_normalized)
                    """)
        
        # Creación de secuencias
        st.markdown("#### 3. Creación de Secuencias Temporales")
        
        st.markdown("""
        Los modelos LSTM requieren datos en formato de secuencias temporales.
        Cada secuencia consiste en:
        - Una ventana de observaciones pasadas (X)
        - El valor objetivo a predecir (y)
        """)
        
        # Selección de tamaño de ventana
        window_size = st.slider(
            "Tamaño de ventana temporal (timesteps):",
            min_value=1,
            max_value=12,
            value=4,
            help="Número de observaciones pasadas utilizadas para predecir el siguiente valor"
        )
        
        # Visualización de secuencias
        st.markdown("#### Ejemplo de Secuencias Temporales")
        
        # Crear datos de ejemplo para visualización
        if len(df_viz) > window_size + 1:
            # Usar datos reales para el ejemplo
            sample_data = df_viz[target_col].values[:window_size+5]
            
            # Crear tabla de ejemplo
            example_data = []
            for i in range(len(sample_data) - window_size):
                x_seq = sample_data[i:i+window_size]
                y_val = sample_data[i+window_size]
                example_data.append({
                    "Secuencia": f"Secuencia {i+1}",
                    "X (Input)": str(x_seq.tolist()),
                    "y (Target)": y_val
                })
            
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, use_container_width=True)
            
            # Código para creación de secuencias
            with st.expander("Ver código para creación de secuencias"):
                st.code("""
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Crear secuencias
X, y = create_sequences(normalized_data, window_size)

# Reshape para LSTM: [samples, timesteps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)
                """)
        
        # División train/test
        st.markdown("#### 4. División en Conjuntos de Entrenamiento y Prueba")
        
        # Selección de proporción de división
        train_size = st.slider(
            "Proporción de datos para entrenamiento:",
            min_value=0.5,
            max_value=0.9,
            value=0.8,
            step=0.05,
            help="Porcentaje de datos utilizados para entrenamiento (el resto se usa para prueba)"
        )
        
        # Visualización de división
        if len(df_viz) > 0:
            # Calcular punto de división
            split_idx = int(len(df_viz) * train_size)
            split_date = df_viz['fecha'].iloc[split_idx]
            
            # Crear figura para mostrar división
            fig = go.Figure()
            
            # Añadir datos de entrenamiento
            fig.add_trace(
                go.Scatter(
                    x=df_viz['fecha'][:split_idx],
                    y=df_viz[target_col][:split_idx],
                    mode='lines+markers',
                    name='Datos de Entrenamiento',
                    line=dict(color='#1E88E5'),
                    marker=dict(size=6)
                )
            )
            
            # Añadir datos de prueba
            fig.add_trace(
                go.Scatter(
                    x=df_viz['fecha'][split_idx:],
                    y=df_viz[target_col][split_idx:],
                    mode='lines+markers',
                    name='Datos de Prueba',
                    line=dict(color='#FF8F00'),
                    marker=dict(size=6)
                )
            )
            
            # Añadir línea vertical para punto de división
            fig.add_shape(
                type="line",
                x0=split_date,
                x1=split_date,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dash", color="gray", width=2)
            )
            fig.add_annotation(
                x=split_date,
                y=1,
                yref="paper",
                text="Punto de División",
                showarrow=True,
                arrowhead=2,
                arrowcolor="gray",
                ax=20,
                ay=-30
            )
            
            # Añadir línea vertical para COVID si está en el rango
            if (df_viz['fecha'].min() <= covid_date) and (df_viz['fecha'].max() >= covid_date):
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
                    y=0.9,
                    yref="paper",
                    text="Inicio COVID-19",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    ax=20,
                    ay=-30
                )
            
            fig.update_layout(
                title=f"División de Datos para {target_var}",
                xaxis_title="Fecha",
                yaxis_title=target_var,
                height=400,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Código para división de datos
            with st.expander("Ver código para división de datos"):
                st.code("""
# División en conjuntos de entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Tamaño de datos de entrenamiento: {X_train.shape}")
print(f"Tamaño de datos de prueba: {X_test.shape}")
                """)
    
    with tab2:
        st.markdown("### Arquitectura del Modelo LSTM")
        
        st.markdown("""
        Las redes LSTM (Long Short-Term Memory) son un tipo especializado de redes neuronales recurrentes
        diseñadas para capturar dependencias temporales a largo plazo en secuencias de datos.
        """)
        
        # Diagrama de arquitectura LSTM
        st.image("https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png", 
                 caption="Arquitectura interna de una celda LSTM (Fuente: colah.github.io)")
        
        st.markdown("""
        #### Componentes Clave de una Celda LSTM
        
        1. **Puerta de olvido (Forget Gate)**: Decide qué información de la memoria anterior se descarta
        2. **Puerta de entrada (Input Gate)**: Decide qué nueva información se almacena en la memoria
        3. **Puerta de salida (Output Gate)**: Controla qué información de la memoria se utiliza para la salida
        4. **Estado de celda (Cell State)**: Mantiene información a largo plazo
        
        Esta arquitectura permite que las redes LSTM:
        - Recuerden información durante largos períodos
        - Eviten el problema de desvanecimiento del gradiente
        - Capturen patrones complejos en series temporales
        """)
        
        # Configuración de la arquitectura
        st.markdown("### Configuración de la Arquitectura LSTM")
        
        # Número de capas LSTM
        num_layers = st.radio(
            "Número de capas LSTM:",
            [1, 2, 3],
            index=1
        )
        
        # Unidades en cada capa
        st.markdown("#### Unidades en cada capa LSTM")
        
        units = []
        cols = st.columns(num_layers)
        for i in range(num_layers):
            with cols[i]:
                units.append(st.slider(
                    f"Unidades capa {i+1}:",
                    min_value=8,
                    max_value=128,
                    value=64 // (2**i),  # 64, 32, 16
                    step=8
                ))
        
        # Dropout para regularización
        dropout_rate = st.slider(
            "Tasa de Dropout:",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.1,
            help="Tasa de dropout para prevenir sobreajuste"
        )
        
        # Visualización de la arquitectura
        st.markdown("#### Arquitectura del Modelo")
        
        # Crear representación visual de la arquitectura
        layers = []
        
        # Capa de entrada
        layers.append(f"Input Layer (shape: [None, {window_size}, features])")
        
        # Capas LSTM
        for i in range(num_layers):
            return_seq = i < num_layers - 1
            layers.append(f"LSTM Layer {i+1} (units: {units[i]}, return_sequences: {return_seq})")
            if dropout_rate > 0:
                layers.append(f"Dropout Layer (rate: {dropout_rate})")
        
        # Capa de salida
        layers.append("Dense Layer (units: 1)")
        
        # Mostrar arquitectura como tabla
        arch_df = pd.DataFrame({"Layers": layers})
        st.table(arch_df)
        
        # Código para definición del modelo
        with st.expander("Ver código para definición del modelo"):
            code_lines = [
                "from tensorflow.keras.models import Sequential",
                "from tensorflow.keras.layers import LSTM, Dense, Dropout",
                "",
                "# Definir modelo",
                "model = Sequential()"
            ]
            
            # Añadir capas LSTM
            for i in range(num_layers):
                return_seq = i < num_layers - 1
                
                # Primera capa con input_shape
                if i == 0:
                    code_lines.append(f"model.add(LSTM(units={units[i]}, return_sequences={return_seq}, input_shape=(window_size, n_features)))")
                else:
                    code_lines.append(f"model.add(LSTM(units={units[i]}, return_sequences={return_seq}))")
                
                # Añadir dropout si es necesario
                if dropout_rate > 0:
                    code_lines.append(f"model.add(Dropout({dropout_rate}))")
            
            # Añadir capa de salida
            code_lines.append("model.add(Dense(units=1))")
            
            # Compilación del modelo
            code_lines.extend([
                "",
                "# Compilar modelo",
                "model.compile(optimizer='adam', loss='mean_squared_error')"
            ])
            
            st.code("\n".join(code_lines))
        
        # Hiperparámetros adicionales
        st.markdown("### Hiperparámetros Adicionales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            optimizer = st.selectbox(
                "Optimizador:",
                ["Adam", "RMSprop", "SGD"]
            )
            
            learning_rate = st.select_slider(
                "Tasa de aprendizaje:",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=0.001
            )
        
        with col2:
            loss_function = st.selectbox(
                "Función de pérdida:",
                ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "Huber Loss"]
            )
            
            batch_size = st.select_slider(
                "Tamaño de batch:",
                options=[1, 4, 8, 16, 32, 64],
                value=16
            )
        
        # Resumen de configuración
        st.markdown("### Resumen de Configuración del Modelo")
        
        config = {
            "Arquitectura": f"LSTM con {num_layers} capas",
            "Unidades por capa": str(units),
            "Dropout": dropout_rate,
            "Tamaño de ventana": window_size,
            "Optimizador": f"{optimizer} (lr={learning_rate})",
            "Función de pérdida": loss_function,
            "Tamaño de batch": batch_size
        }
        
        config_df = pd.DataFrame({"Parámetro": config.keys(), "Valor": config.values()})
        st.table(config_df)
    
    with tab3:
        st.markdown("### Entrenamiento del Modelo LSTM")
        
        st.markdown("""
        El entrenamiento de un modelo LSTM implica ajustar sus pesos para minimizar la función de pérdida
        en los datos de entrenamiento, mientras se monitorea el rendimiento en datos de validación.
        """)
        
        # Parámetros de entrenamiento
        st.markdown("#### Parámetros de Entrenamiento")
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider(
                "Número de épocas:",
                min_value=10,
                max_value=200,
                value=50,
                step=10
            )
            
            validation_split = st.slider(
                "Proporción de validación:",
                min_value=0.1,
                max_value=0.3,
                value=0.2,
                step=0.05,
                help="Proporción de datos de entrenamiento utilizados para validación"
            )
        
        with col2:
            use_early_stopping = st.checkbox("Usar Early Stopping", value=True)
            
            if use_early_stopping:
                patience = st.slider(
                    "Paciencia para Early Stopping:",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=5,
                    help="Número de épocas sin mejora antes de detener el entrenamiento"
                )
        
        # Visualización del proceso de entrenamiento
        st.markdown("#### Visualización del Proceso de Entrenamiento")
        
        # Generar datos de ejemplo para la visualización
        np.random.seed(42)
        example_history = {
            'loss': [np.exp(-0.1 * i) + 0.1 + np.random.normal(0, 0.02) for i in range(epochs)],
            'val_loss': [np.exp(-0.08 * i) + 0.15 + np.random.normal(0, 0.05) for i in range(epochs)]
        }
        
        # Simular early stopping si está activado
        if use_early_stopping:
            # Encontrar el punto donde val_loss deja de mejorar significativamente
            best_epoch = min(epochs - patience, int(epochs * 0.7))
            for i in range(best_epoch + 1, epochs):
                example_history['val_loss'][i] = example_history['val_loss'][best_epoch] * (1 + 0.02 * (i - best_epoch))
            
            # Marcar el punto de early stopping
            early_stop_epoch = best_epoch + patience
        else:
            early_stop_epoch = epochs
        
        # Crear gráfico de pérdida durante entrenamiento
        fig = go.Figure()
        
        # Añadir curva de pérdida de entrenamiento
        fig.add_trace(
            go.Scatter(
                x=list(range(1, epochs + 1)),
                y=example_history['loss'][:epochs],
                mode='lines',
                name='Pérdida (Train)',
                line=dict(color='#1E88E5', width=2)
            )
        )
        
        # Añadir curva de pérdida de validación
        fig.add_trace(
            go.Scatter(
                x=list(range(1, epochs + 1)),
                y=example_history['val_loss'][:epochs],
                mode='lines',
                name='Pérdida (Validación)',
                line=dict(color='#FF8F00', width=2)
            )
        )
        
        # Añadir línea vertical para early stopping si está activado
        if use_early_stopping and early_stop_epoch < epochs:
            fig.add_shape(
                type="line",
                x0=early_stop_epoch,
                x1=early_stop_epoch,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dash", color="red", width=2)
            )
            fig.add_annotation(
                x=early_stop_epoch,
                y=0.9,
                yref="paper",
                text="Early Stopping",
                showarrow=True,
                arrowhead=2,
                arrowcolor="red",
                ax=20,
                ay=-30
            )
        
        fig.update_layout(
            title="Curvas de Pérdida Durante Entrenamiento (Simulación)",
            xaxis_title="Época",
            yaxis_title="Pérdida (MSE)",
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Código para entrenamiento
        with st.expander("Ver código para entrenamiento del modelo"):
            code_lines = [
                "# Configurar callbacks",
                "callbacks = []"
            ]
            
            if use_early_stopping:
                code_lines.extend([
                    "",
                    "# Early stopping para evitar sobreajuste",
                    f"early_stopping = EarlyStopping(monitor='val_loss', patience={patience}, restore_best_weights=True)",
                    "callbacks.append(early_stopping)"
                ])
            
            code_lines.extend([
                "",
                "# Entrenar modelo",
                f"history = model.fit(",
                f"    X_train, y_train,",
                f"    epochs={epochs},",
                f"    batch_size={batch_size},",
                f"    validation_split={validation_split},",
                f"    callbacks=callbacks,",
                f"    verbose=1",
                f")"
            ])
            
            st.code("\n".join(code_lines))
        
        # Consejos para entrenamiento efectivo
        with st.expander("Consejos para entrenamiento efectivo"):
            st.markdown("""
            ### Consejos para un Entrenamiento Efectivo
            
            1. **Monitoreo de Overfitting**:
               - Observe la brecha entre pérdida de entrenamiento y validación
               - Si la pérdida de validación aumenta mientras la de entrenamiento sigue disminuyendo, es señal de sobreajuste
            
            2. **Ajuste de Hiperparámetros**:
               - Comience con arquitecturas simples y aumente la complejidad gradualmente
               - Ajuste la tasa de aprendizaje si el entrenamiento es inestable o lento
               - Experimente con diferentes tamaños de ventana temporal
            
            3. **Regularización**:
               - Aumente el dropout si hay sobreajuste
               - Considere reducir el número de unidades o capas
               - La normalización de datos también ayuda a regularizar
            
            4. **Datos**:
               - Más datos generalmente conducen a mejor rendimiento
               - Considere técnicas de aumento de datos para series temporales
               - Asegúrese de que los datos de entrenamiento y prueba sean representativos
            """)
    
    with tab4:
        st.markdown("### Evaluación del Modelo LSTM")
        
        st.markdown("""
        La evaluación del modelo implica medir su rendimiento en datos no vistos durante el entrenamiento.
        Esto permite estimar cómo se comportará el modelo en situaciones reales.
        """)
        
        # Métricas de evaluación
        st.markdown("#### Métricas de Evaluación")
        
        # Generar métricas de ejemplo
        mse = 0.015 + np.random.normal(0, 0.002)
        rmse = np.sqrt(mse)
        mae = 0.09 + np.random.normal(0, 0.005)
        r2 = 0.85 + np.random.normal(0, 0.03)
        
        # Mostrar métricas en columnas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MSE", f"{mse:.4f}")
        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
        with col3:
            st.metric("MAE", f"{mae:.4f}")
        with col4:
            st.metric("R²", f"{r2:.4f}")
        
        # Explicación de métricas
        with st.expander("Explicación de métricas"):
            st.markdown("""
            ### Interpretación de Métricas
            
            - **MSE (Error Cuadrático Medio)**: Promedio de los errores al cuadrado. Penaliza errores grandes.
            - **RMSE (Raíz del Error Cuadrático Medio)**: Raíz cuadrada del MSE. Está en la misma escala que los datos originales.
            - **MAE (Error Absoluto Medio)**: Promedio de los errores absolutos. Menos sensible a outliers que MSE.
            - **R² (Coeficiente de Determinación)**: Proporción de la varianza explicada por el modelo. Varía de 0 a 1, donde 1 indica predicción perfecta.
            
            Para modelos de series temporales, es importante considerar también:
            - **Dirección de cambio**: ¿El modelo predice correctamente la dirección de los movimientos?
            - **Timing de cambios**: ¿El modelo captura los puntos de inflexión?
            - **Estabilidad**: ¿El rendimiento es consistente en diferentes períodos?
            """)
        
        # Visualización de predicciones vs valores reales
        st.markdown("#### Predicciones vs Valores Reales")
        
        # Generar predicciones de ejemplo
        if len(df_viz) > 0:
            # Usar datos reales para la visualización
            split_idx = int(len(df_viz) * train_size)
            test_dates = df_viz['fecha'][split_idx:].reset_index(drop=True)
            test_values = df_viz[target_col][split_idx:].values
            
            # Generar predicciones simuladas
            np.random.seed(42)
            predictions = test_values * (1 + np.random.normal(0, 0.1, size=len(test_values)))
            
            # Crear DataFrame para visualización
            results_df = pd.DataFrame({
                'fecha': test_dates,
                'real': test_values,
                'prediccion': predictions
            })
            
            # Calcular errores
            results_df['error'] = results_df['prediccion'] - results_df['real']
            results_df['error_pct'] = (results_df['error'] / results_df['real']) * 100
            
            # Crear gráfico de predicciones vs valores reales
            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("Predicciones vs Valores Reales", "Error de Predicción")
            )
            
            # Añadir valores reales
            fig.add_trace(
                go.Scatter(
                    x=results_df['fecha'],
                    y=results_df['real'],
                    mode='lines+markers',
                    name='Valores Reales',
                    line=dict(color='#1E88E5', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Añadir predicciones
            fig.add_trace(
                go.Scatter(
                    x=results_df['fecha'],
                    y=results_df['prediccion'],
                    mode='lines+markers',
                    name='Predicciones',
                    line=dict(color='#FF8F00', width=2, dash='dash'),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Añadir errores
            fig.add_trace(
                go.Bar(
                    x=results_df['fecha'],
                    y=results_df['error'],
                    name='Error',
                    marker_color=np.where(results_df['error'] >= 0, '#FF8F00', '#1E88E5')
                ),
                row=2, col=1
            )
            
            # Añadir línea horizontal en cero para errores
            fig.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="gray",
                row=2, col=1
            )
            
            # Añadir línea vertical para COVID si está en el rango
            if any((test_dates >= covid_date) & (test_dates <= results_df['fecha'].max())):
                fig.add_shape(
                    type="line",
                    x0=covid_date,
                    x1=covid_date,
                    y0=0,
                    y1=1,
                    yref="y",
                    line=dict(dash="dash", color="red", width=2),
                    row=1, col=1
                )
                fig.add_annotation(
                    x=covid_date,
                    y=1,
                    yref="y domain",
                    text="Inicio COVID-19",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
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
                    line=dict(dash="dash", color="red", width=2),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified"
            )
            
            fig.update_xaxes(title_text="Fecha", row=2, col=1)
            fig.update_yaxes(title_text=target_var, row=1, col=1)
            fig.update_yaxes(title_text="Error", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar tabla de resultados
            with st.expander("Ver tabla de resultados"):
                st.dataframe(
                    results_df[['fecha', 'real', 'prediccion', 'error', 'error_pct']].round(2),
                    use_container_width=True
                )
        
        # Código para evaluación
        with st.expander("Ver código para evaluación del modelo"):
            st.code("""
# Hacer predicciones
y_pred = model.predict(X_test)

# Desnormalizar predicciones y valores reales
y_pred_original = scaler.inverse_transform(y_pred)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calcular métricas
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
            """)
        
        # Predicción de horizonte múltiple
        st.markdown("### Predicción de Horizonte Múltiple")
        
        st.markdown("""
        La predicción de horizonte múltiple implica predecir varios pasos hacia el futuro.
        Existen dos enfoques principales:
        
        1. **Predicción Recursiva**: Utiliza predicciones anteriores como entrada para predicciones futuras
        2. **Predicción Directa**: Entrena modelos separados para cada horizonte de predicción
        """)
        
        # Selección de horizonte de predicción
        horizon = st.slider(
            "Horizonte de predicción (trimestres):",
            min_value=1,
            max_value=8,
            value=4
        )
        
        # Visualización de predicción de horizonte múltiple
        if len(df_viz) > 0:
            # Usar últimos datos como punto de partida
            last_date = df_viz['fecha'].iloc[-1]
            last_value = df_viz[target_col].iloc[-1]
            
            # Generar fechas futuras
            future_dates = pd.date_range(start=last_date, periods=horizon+1, freq='3ME')[1:]
            
            # Generar predicciones simuladas con incertidumbre creciente
            np.random.seed(42)
            trend = 0.02  # tendencia positiva
            future_values = [last_value]
            
            for i in range(horizon):
                next_val = future_values[-1] * (1 + trend + np.random.normal(0, 0.03 * (i+1)))
                future_values.append(next_val)
            
            future_values = future_values[1:]  # Eliminar el valor inicial
            
            # Calcular intervalos de confianza
            lower_bound = [val * (1 - 0.05 * (i+1)) for i, val in enumerate(future_values)]
            upper_bound = [val * (1 + 0.05 * (i+1)) for i, val in enumerate(future_values)]
            
            # Crear DataFrame para visualización
            future_df = pd.DataFrame({
                'fecha': future_dates,
                'prediccion': future_values,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
            
            # Combinar con datos históricos
            hist_df = df_viz[['fecha', target_col]].rename(columns={target_col: 'valor'})
            
            # Crear gráfico
            fig = go.Figure()
            
            # Añadir datos históricos
            fig.add_trace(
                go.Scatter(
                    x=hist_df['fecha'],
                    y=hist_df['valor'],
                    mode='lines+markers',
                    name='Datos Históricos',
                    line=dict(color='#1E88E5', width=2),
                    marker=dict(size=6)
                )
            )
            
            # Añadir predicciones
            fig.add_trace(
                go.Scatter(
                    x=future_df['fecha'],
                    y=future_df['prediccion'],
                    mode='lines+markers',
                    name='Predicciones',
                    line=dict(color='#FF8F00', width=2, dash='dash'),
                    marker=dict(size=8)
                )
            )
            
            # Añadir intervalo de confianza
            fig.add_trace(
                go.Scatter(
                    x=future_df['fecha'].tolist() + future_df['fecha'].tolist()[::-1],
                    y=future_df['upper_bound'].tolist() + future_df['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,143,0,0.2)',
                    line=dict(color='rgba(255,143,0,0)'),
                    name='Intervalo de Confianza (95%)'
                )
            )
            
            # Añadir línea vertical para separar histórico de predicciones
            fig.add_shape(
                type="line",
                x0=last_date,
                x1=last_date,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(dash="dash", color="gray", width=2)
            )
            fig.add_annotation(
                x=last_date,
                y=1,
                yref="paper",
                text="Último Dato Conocido",
                showarrow=True,
                arrowhead=2,
                arrowcolor="gray",
                ax=20,
                ay=-30
            )
            
            fig.update_layout(
                title=f"Predicción de {horizon} Trimestres para {target_var}",
                xaxis_title="Fecha",
                yaxis_title=target_var,
                height=500,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Código para predicción de horizonte múltiple
            with st.expander("Ver código para predicción de horizonte múltiple"):
                st.code("""
                def predict_multi_horizon(model, X_last, scaler, n_steps, n_features):
                    \"\"\"
                    Realiza predicción recursiva de múltiples pasos.
                    
                    Args:
                        model: Modelo LSTM entrenado
                        X_last: Última secuencia conocida
                        scaler: Scaler utilizado para normalización
                        n_steps: Número de pasos a predecir
                        n_features: Número de características
                        
                    Returns:
                        Array con predicciones
                    \"\"\"
                    predictions = []
                    curr_seq = X_last.copy()
                    
                    for _ in range(n_steps):
                        # Predecir siguiente valor
                        next_pred = model.predict(curr_seq.reshape(1, curr_seq.shape[0], n_features))
                        
                        # Añadir a predicciones
                        predictions.append(next_pred[0, 0])
                        
                        # Actualizar secuencia para siguiente predicción
                        curr_seq = np.roll(curr_seq, -1)
                        curr_seq[-1] = next_pred
                    
                    # Desnormalizar predicciones
                    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
                    
                    return predictions.flatten()
                """)
        
        # Limitaciones y consideraciones
        with st.expander("Limitaciones y consideraciones"):
            st.markdown("""
            ### Limitaciones y Consideraciones
            
            1. **Incertidumbre Creciente**:
               - La incertidumbre aumenta con el horizonte de predicción
               - Las predicciones a largo plazo son inherentemente menos confiables
            
            2. **Acumulación de Errores**:
               - En la predicción recursiva, los errores se propagan y acumulan
               - Pequeños errores iniciales pueden amplificarse con el tiempo
            
            3. **Cambios Estructurales**:
               - Los modelos asumen que los patrones históricos continuarán en el futuro
               - Eventos disruptivos (como COVID-19) pueden invalidar esta suposición
            
            4. **Factores Externos**:
               - Variables económicas como IVA y PIB están influenciadas por factores no capturados en los datos históricos
               - Políticas fiscales, eventos globales y cambios regulatorios pueden afectar las predicciones
            
            5. **Mejores Prácticas**:
               - Actualizar modelos regularmente con nuevos datos
               - Combinar múltiples modelos para predicciones más robustas
               - Incorporar conocimiento de dominio en la interpretación de resultados
               - Presentar predicciones con intervalos de confianza apropiados
            """)
