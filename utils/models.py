import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_lstm_model(input_shape, lstm_units=[64, 32], dropout_rate=0.2):
    """
    Crea un modelo LSTM para predicción de series temporales.
    
    Args:
        input_shape: Tupla con forma de entrada (timesteps, features)
        lstm_units: Lista con número de unidades en cada capa LSTM
        dropout_rate: Tasa de dropout para regularización
    
    Returns:
        Modelo LSTM compilado
    """
    model = Sequential()
    
    # Añadir capas LSTM
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        
        if i == 0:
            model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units=units, return_sequences=return_sequences))
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Añadir capa de salida
    model.add(Dense(units=1))
    
    # Compilar modelo
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, 
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
        Diccionario con métricas de evaluación y predicciones
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
        predictions.append(next_pred[0, 0])
        
        # Actualizar secuencia para siguiente predicción
        if len(curr_sequence.shape) > 1:
            # Para datos multivariados
            new_seq = np.append(curr_sequence[1:], [[next_pred[0, 0]]], axis=0)
            curr_sequence = new_seq
        else:
            # Para datos univariados
            new_seq = np.append(curr_sequence[1:], next_pred[0, 0])
            curr_sequence = new_seq
    
    # Convertir a array
    predictions = np.array(predictions).reshape(-1, 1)
    
    # Desnormalizar si se proporciona scaler
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()

def save_model(model, model_path):
    """
    Guarda un modelo entrenado.
    
    Args:
        model: Modelo entrenado
        model_path: Ruta donde guardar el modelo
    """
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

def load_saved_model(model_path):
    """
    Carga un modelo guardado.
    
    Args:
        model_path: Ruta al modelo guardado
    
    Returns:
        Modelo cargado
    """
    return load_model(model_path)

def calculate_prediction_intervals(predictions, error_std, confidence=0.95):
    """
    Calcula intervalos de predicción.
    
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

def compare_models(models_dict, X_test, y_test, scaler=None):
    """
    Compara múltiples modelos.
    
    Args:
        models_dict: Diccionario con nombres de modelos y modelos entrenados
        X_test: Datos de prueba
        y_test: Valores objetivo de prueba
        scaler: Scaler utilizado para normalización (opcional)
    
    Returns:
        DataFrame con métricas comparativas
    """
    results = []
    
    for name, model in models_dict.items():
        metrics, _ = evaluate_model(model, X_test, y_test, scaler)
        metrics['modelo'] = name
        results.append(metrics)
    
    return pd.DataFrame(results)

def create_ensemble_prediction(models_list, X, weights=None):
    """
    Crea una predicción de conjunto (ensemble) a partir de múltiples modelos.
    
    Args:
        models_list: Lista de modelos entrenados
        X: Datos de entrada
        weights: Pesos para cada modelo (opcional)
    
    Returns:
        Predicciones de conjunto
    """
    # Si no se proporcionan pesos, usar pesos iguales
    if weights is None:
        weights = [1/len(models_list)] * len(models_list)
    
    # Normalizar pesos
    weights = np.array(weights) / sum(weights)
    
    # Obtener predicciones de cada modelo
    predictions = []
    for model in models_list:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Calcular predicción ponderada
    ensemble_pred = np.zeros_like(predictions[0])
    for i, pred in enumerate(predictions):
        ensemble_pred += pred * weights[i]
    
    return ensemble_pred
