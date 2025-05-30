# Documentación del Tablero Streamlit para Análisis de Series Temporales con LSTM

## Descripción General

Este proyecto implementa un tablero interactivo con Streamlit para el análisis y modelado de series temporales económicas utilizando redes neuronales LSTM. El tablero está diseñado para ser utilizado en una clase de maestría de Ciencia de Datos, permitiendo explorar, visualizar y modelar datos de IVA y PIB a través de una interfaz moderna y didáctica.

## Estructura del Proyecto

```
proyecto_streamlit/
│
├── app.py                  # Aplicación principal de Streamlit
├── data/                   # Carpeta para archivos de datos
│   ├── datos_procesados.csv
│   └── datos_iva_pib.csv
│
├── modules/                # Módulos funcionales del tablero
│   ├── intro.py            # Introducción y contexto
│   ├── exploracion.py      # Exploración inicial de datos
│   ├── preprocesamiento.py # Preprocesamiento de datos
│   ├── analisis.py         # Análisis exploratorio avanzado
│   ├── modelado.py         # Modelado LSTM
│   └── resultados.py       # Resultados y conclusiones
│
└── utils/                  # Utilidades reutilizables
    ├── visualization.py    # Funciones de visualización
    ├── data_processing.py  # Funciones de procesamiento de datos
    └── models.py           # Funciones para modelos LSTM
```

## Requisitos

Para ejecutar este proyecto, necesitará tener instalado:

1. Python 3.8 o superior
2. Las siguientes bibliotecas de Python:
   - streamlit
   - pandas
   - numpy
   - matplotlib
   - plotly
   - seaborn
   - scikit-learn
   - tensorflow
   - keras

## Instalación de Dependencias

Puede instalar todas las dependencias necesarias ejecutando el siguiente comando en la terminal de VSCode:

```bash
pip install streamlit pandas numpy matplotlib plotly seaborn scikit-learn tensorflow
```

## Ejecución del Tablero en VSCode

Para ejecutar el tablero Streamlit en VSCode, siga estos pasos:

1. Abra VSCode y navegue hasta la carpeta del proyecto.
2. Asegúrese de que los archivos de datos estén en la carpeta `data/`.
3. Abra una terminal en VSCode (Terminal > New Terminal).
4. Ejecute el siguiente comando:

```bash
streamlit run app.py
```

5. Streamlit iniciará un servidor local y abrirá automáticamente el tablero en su navegador predeterminado.
6. Si no se abre automáticamente, puede acceder al tablero en la URL que se muestra en la terminal (generalmente http://localhost:8501).

## Estructura de Datos

El tablero espera dos archivos CSV en la carpeta `data/`:

1. `datos_procesados.csv`: Datos preprocesados con las siguientes columnas:
   - TRIMESTRE: Identificador de trimestre (formato: YYYY-QN)
   - IVA_TOTAL: Valores de IVA total
   - PIB: Valores de PIB
   - IVA_pct_change: Cambio porcentual en IVA (opcional)
   - PIB_pct_change: Cambio porcentual en PIB (opcional)

2. `datos_iva_pib.csv`: Datos originales con estructura similar.

Si los archivos no están disponibles, el tablero generará datos de ejemplo para demostración.

## Navegación por el Tablero

El tablero está organizado en seis secciones principales, accesibles desde el menú lateral:

1. **Introducción**: Contexto y objetivos del análisis.
2. **Exploración de Datos**: Visualización inicial de los datos disponibles.
3. **Preprocesamiento**: Técnicas de limpieza y transformación de datos.
4. **Análisis Exploratorio**: Análisis detallado de patrones y relaciones.
5. **Modelado LSTM**: Configuración, entrenamiento y evaluación de modelos LSTM.
6. **Resultados y Conclusiones**: Resumen de hallazgos y recomendaciones.

## Personalización y Desarrollo

### Modificación de Módulos

Cada sección del tablero está implementada como un módulo independiente en la carpeta `modules/`. Para modificar una sección específica:

1. Abra el archivo correspondiente en VSCode (por ejemplo, `modules/analisis.py`).
2. Realice los cambios deseados en la función `show()`.
3. Guarde el archivo y recargue el tablero en el navegador.

### Adición de Nuevas Funcionalidades

Para añadir nuevas funcionalidades:

1. Cree un nuevo módulo en la carpeta `modules/` siguiendo el patrón existente.
2. Implemente la función `show()` que reciba los datos necesarios.
3. Actualice `app.py` para incluir el nuevo módulo en el menú y la lógica de navegación.

### Utilización de Utilidades

El proyecto incluye tres archivos de utilidades en la carpeta `utils/`:

1. `visualization.py`: Funciones para crear visualizaciones con Plotly.
2. `data_processing.py`: Funciones para procesamiento y transformación de datos.
3. `models.py`: Funciones para crear, entrenar y evaluar modelos LSTM.

Para utilizar estas utilidades en sus módulos:

```python
from utils.visualization import create_time_series_plot
from utils.data_processing import normalize_data
from utils.models import create_lstm_model
```

## Consejos para Desarrollo en VSCode

### Extensiones Recomendadas

Para mejorar su experiencia de desarrollo, considere instalar estas extensiones de VSCode:

1. **Python**: Soporte para Python, incluyendo IntelliSense y depuración.
2. **Pylance**: Servidor de lenguaje para Python con mejor autocompletado.
3. **Jupyter**: Soporte para notebooks de Jupyter.
4. **Rainbow CSV**: Facilita la visualización de archivos CSV.
5. **Prettier**: Formateador de código para mantener consistencia.

### Configuración del Entorno Virtual

Es recomendable utilizar un entorno virtual para este proyecto:

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual (Windows)
venv\Scripts\activate

# Activar entorno virtual (macOS/Linux)
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### Depuración en VSCode

Para depurar el tablero Streamlit en VSCode:

1. Cree un archivo `.vscode/launch.json` con la siguiente configuración:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Streamlit",
            "type": "python",
            "request": "launch",
            "module": "streamlit",
            "args": [
                "run",
                "app.py"
            ],
            "justMyCode": true
        }
    ]
}
```

2. Establezca puntos de interrupción en su código.
3. Inicie la depuración desde la pestaña "Run and Debug" de VSCode.

## Desarrollo Colaborativo

Para trabajar en equipo con este proyecto:

1. Utilice control de versiones (Git) para gestionar cambios.
2. Divida el trabajo por módulos o funcionalidades.
3. Establezca convenciones de código claras.
4. Realice revisiones de código antes de integrar cambios.
5. Documente todas las funciones y clases con docstrings.

## Solución de Problemas Comunes

### El tablero no se inicia

- Verifique que todas las dependencias estén instaladas.
- Compruebe que está ejecutando el comando desde la carpeta correcta.
- Verifique que no haya errores de sintaxis en el código.

### Errores de importación

- Asegúrese de que la estructura de carpetas sea correcta.
- Verifique que los nombres de los módulos coincidan con los importados.
- Compruebe que el archivo `__init__.py` esté presente en las carpetas de módulos.

### Problemas con TensorFlow

- Verifique que tiene instalada la versión correcta de TensorFlow.
- En algunos sistemas, puede ser necesario instalar TensorFlow con GPU para mejor rendimiento.

## Recursos Adicionales

- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Documentación de Plotly](https://plotly.com/python/)
- [Documentación de TensorFlow](https://www.tensorflow.org/api_docs)
- [Tutorial de Series Temporales con LSTM](https://www.tensorflow.org/tutorials/structured_data/time_series)
