import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def load_data(file_path, date_col=None, date_format=None, index_col=None):
    """
    Carga datos desde un archivo CSV y realiza conversiones de fecha.
    
    Args:
        file_path: Ruta al archivo CSV
        date_col: Nombre de la columna de fecha (opcional)
        date_format: Formato de fecha (opcional)
        index_col: Columna a usar como índice (opcional)
    
    Returns:
        DataFrame de pandas con los datos cargados
    """
    # Cargar datos
    if index_col is not None:
        df = pd.read_csv(file_path, index_col=index_col)
    else:
        df = pd.read_csv(file_path)
    
    # Convertir columna de fecha si se especifica
    if date_col is not None:
        if date_format is not None:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        else:
            df[date_col] = pd.to_datetime(df[date_col])
    
    return df

def preprocess_time_series(df, target_col, date_col=None, fill_method='ffill', add_pct_change=True):
    """
    Preprocesa datos de series temporales.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna objetivo
        date_col: Nombre de la columna de fecha (opcional)
        fill_method: Método para rellenar valores faltantes ('ffill', 'bfill', 'interpolate')
        add_pct_change: Si se debe añadir columna de cambio porcentual
    
    Returns:
        DataFrame preprocesado
    """
    # Crear copia para no modificar el original
    processed_df = df.copy()
    
    # Ordenar por fecha si se proporciona
    if date_col is not None:
        processed_df = processed_df.sort_values(date_col)
    
    # Rellenar valores faltantes
    if fill_method == 'ffill':
        processed_df = processed_df.fillna(method='ffill')
    elif fill_method == 'bfill':
        processed_df = processed_df.fillna(method='bfill')
    elif fill_method == 'interpolate':
        processed_df = processed_df.interpolate(method='linear')
    
    # Añadir cambio porcentual si se solicita
    if add_pct_change and target_col in processed_df.columns:
        processed_df[f'{target_col}_pct_change'] = processed_df[target_col].pct_change() * 100
    
    return processed_df

def normalize_data(data, method='minmax', feature_range=(0, 1)):
    """
    Normaliza datos numéricos.
    
    Args:
        data: Array o DataFrame con datos a normalizar
        method: Método de normalización ('minmax' o 'zscore')
        feature_range: Rango para normalización MinMax
    
    Returns:
        Datos normalizados y scaler utilizado
    """
    # Reshape si es necesario
    if isinstance(data, pd.Series):
        data_reshaped = data.values.reshape(-1, 1)
    elif isinstance(data, np.ndarray) and data.ndim == 1:
        data_reshaped = data.reshape(-1, 1)
    else:
        data_reshaped = data
    
    # Aplicar normalización
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    else:  # zscore
        scaler = StandardScaler()
    
    normalized_data = scaler.fit_transform(data_reshaped)
    
    return normalized_data, scaler

def create_sequences(data, window_size, target_idx=-1, flatten=False):
    """
    Crea secuencias para entrenamiento de modelos de series temporales.
    
    Args:
        data: Array con datos normalizados
        window_size: Tamaño de ventana (número de timesteps)
        target_idx: Índice de la variable objetivo en caso de datos multivariados
        flatten: Si se deben aplanar las secuencias (para modelos no RNN)
    
    Returns:
        X (secuencias de entrada) e y (valores objetivo)
    """
    X, y = [], []
    
    # Asegurar que data sea un array numpy
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.values
    
    # Crear secuencias
    for i in range(len(data) - window_size):
        # Secuencia de entrada
        seq = data[i:i+window_size]
        X.append(seq)
        
        # Valor objetivo (siguiente valor después de la secuencia)
        if data.ndim > 1 and target_idx is not None:
            y.append(data[i+window_size, target_idx])
        else:
            y.append(data[i+window_size])
    
    # Convertir a arrays numpy
    X = np.array(X)
    y = np.array(y)
    
    # Aplanar si se solicita
    if flatten:
        X = X.reshape(X.shape[0], -1)
    
    return X, y

def train_test_split_time_series(X, y, train_size=0.8):
    """
    Divide datos de series temporales en conjuntos de entrenamiento y prueba.
    
    Args:
        X: Secuencias de entrada
        y: Valores objetivo
        train_size: Proporción de datos para entrenamiento (0-1)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Calcular punto de división
    split_idx = int(len(X) * train_size)
    
    # Dividir datos
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def build_lstm_model(input_shape, units=[64, 32], dropout_rate=0.2, output_units=1):
    """
    Construye un modelo LSTM para series temporales.
    
    Args:
        input_shape: Forma de los datos de entrada (timesteps, features)
        units: Lista con número de unidades en cada capa LSTM
        dropout_rate: Tasa de dropout para regularización
        output_units: Número de unidades en la capa de salida
    
    Returns:
        Modelo LSTM compilado
    """
    model = Sequential()
    
    # Añadir capas LSTM
    for i, unit in enumerate(units):
        return_sequences = i < len(units) - 1  # True excepto para la última capa LSTM
        
        if i == 0:
            model.add(LSTM(units=unit, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units=unit, return_sequences=return_sequences))
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Añadir capa de salida
    model.add(Dense(units=output_units))
    
    # Compilar modelo
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, 
                    early_stopping=True, patience=10, verbose=1):
    """
    Entrena un modelo LSTM.
    
    Args:
        model: Modelo LSTM compilado
        X_train: Datos de entrenamiento
        y_train: Valores objetivo de entrenamiento
        epochs: Número de épocas
        batch_size: Tamaño de batch
        validation_split: Proporción de datos para validación
        early_stopping: Si se debe usar early stopping
        patience: Paciencia para early stopping
        verbose: Nivel de verbosidad (0, 1, 2)
    
    Returns:
        Historial de entrenamiento y modelo entrenado
    """
    callbacks = []
    
    # Añadir early stopping si está habilitado
    if early_stopping:
        es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        callbacks.append(es)
    
    # Entrenar modelo
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return history, model

def evaluate_model(model, X_test, y_test, scaler=None):
    """
    Evalúa un modelo entrenado.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Valores objetivo de prueba
        scaler: Scaler utilizado para normalización (opcional)
    
    Returns:
        Diccionario con métricas de evaluación
    """
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Desnormalizar si se proporciona scaler
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred)
        
        # Reshape y_test si es necesario
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
        
        y_test = scaler.inverse_transform(y_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Crear diccionario de métricas
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    return metrics, y_pred

def predict_future(model, last_sequence, n_steps, scaler=None):
    """
    Realiza predicciones futuras utilizando un modelo entrenado.
    
    Args:
        model: Modelo entrenado
        last_sequence: Última secuencia conocida
        n_steps: Número de pasos futuros a predecir
        scaler: Scaler utilizado para normalización (opcional)
    
    Returns:
        Array con predicciones futuras
    """
    # Hacer copia de la última secuencia
    curr_sequence = last_sequence.copy()
    
    # Lista para almacenar predicciones
    predictions = []
    
    # Predecir n_steps hacia el futuro
    for _ in range(n_steps):
        # Reshape para predicción
        curr_sequence_reshaped = curr_sequence.reshape(1, curr_sequence.shape[0], curr_sequence.shape[1] 
                                                      if len(curr_sequence.shape) > 1 else 1)
        
        # Predecir siguiente valor
        next_pred = model.predict(curr_sequence_reshaped)
        
        # Añadir predicción a la lista
        predictions.append(next_pred[0])
        
        # Actualizar secuencia para siguiente predicción
        if len(curr_sequence.shape) > 1:
            # Para datos multivariados
            new_seq = np.append(curr_sequence[1:], [[next_pred[0]]], axis=0)
            curr_sequence = new_seq
        else:
            # Para datos univariados
            new_seq = np.append(curr_sequence[1:], next_pred[0])
            curr_sequence = new_seq
    
    # Convertir a array
    predictions = np.array(predictions)
    
    # Desnormalizar si se proporciona scaler
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
    
    return predictions

def calculate_forecast_intervals(predictions, error_std, confidence=0.95):
    """
    Calcula intervalos de confianza para predicciones.
    
    Args:
        predictions: Array con predicciones
        error_std: Desviación estándar del error
        confidence: Nivel de confianza (0-1)
    
    Returns:
        Arrays con límites inferior y superior
    """
    from scipy import stats
    
    # Calcular z-score para el nivel de confianza
    z = stats.norm.ppf((1 + confidence) / 2)
    
    # Calcular intervalos
    lower_bound = predictions - z * error_std
    upper_bound = predictions + z * error_std
    
    return lower_bound, upper_bound

def detect_anomalies(data, window_size=10, threshold=2.0):
    """
    Detecta anomalías en series temporales usando desviación estándar móvil.
    
    Args:
        data: Serie temporal
        window_size: Tamaño de ventana para cálculo de estadísticas
        threshold: Umbral en número de desviaciones estándar
    
    Returns:
        Índices de anomalías y puntuaciones Z
    """
    # Convertir a array numpy si es necesario
    if isinstance(data, pd.Series):
        data = data.values
    
    # Calcular media y desviación estándar móviles
    rolling_mean = np.array([np.mean(data[max(0, i-window_size):i]) 
                            for i in range(1, len(data)+1)])
    rolling_std = np.array([np.std(data[max(0, i-window_size):i]) 
                           for i in range(1, len(data)+1)])
    
    # Evitar división por cero
    rolling_std[rolling_std == 0] = np.mean(rolling_std[rolling_std > 0])
    
    # Calcular puntuaciones Z
    z_scores = (data - rolling_mean) / rolling_std
    
    # Identificar anomalías
    anomalies_idx = np.where(np.abs(z_scores) > threshold)[0]
    
    return anomalies_idx, z_scores
