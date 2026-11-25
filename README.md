# Clasificación de radiografías de tórax · Proyecto final

**Grupo 2 · (Juan)⁴ · Laboratorio de Inteligencia Artificial**

Aplicación académica en Streamlit que utiliza MobileNetV2 para clasificar radiografías del conjunto de práctica como **NORMAL** o **PNEUMONIA**. La entrega contiene el modelo entrenado, la interfaz, las imágenes del test y los resultados del experimento. El procesamiento se realiza en el computador que ejecuta la aplicación.

Abre **GUIA_INICIO.html** para consultar los pasos de uso. El guion breve para presentar el proyecto está en **GUIA_EXPOSICION.txt**.

## 1. Elegir cómo instalar esta entrega

### Si ya tienes la versión anterior funcionando

Puedes conservar el entorno que ya instalaste:

1. Detén Streamlit con `Ctrl+C` en su terminal y conserva el ZIP anterior como respaldo.
2. Extrae `Neumonia_Proyecto_Final.zip` en una ubicación temporal.
3. Copia el contenido de la nueva carpeta `neumonia_app` sobre la carpeta `neumonia_app` que ya utilizabas. Acepta reemplazar los archivos del proyecto.
4. **Conserva la carpeta `.venv` en su ubicación actual.** Este ZIP no la incluye ni necesita reemplazarla.
5. Abre `INICIAR_PAGINA.bat` o ejecuta el comando de inicio de la sección 2.

Así actualizas la página y las imágenes conservando las bibliotecas ya instaladas. Las dependencias de esta entrega son las mismas de la versión que ya funcionó en el computador de Esteban.

### Si usarás otra carpeta o el computador de un compañero

1. Instala **Python 3.12 de 64 bits** y extrae el ZIP completo.
2. Abre la carpeta `neumonia_app`.
3. Ejecuta **INSTALAR_ENTORNO.bat**. Crea el entorno `.venv` e instala las dependencias desde sus repositorios oficiales; necesita Internet y puede tardar varios minutos.
4. Cuando indique que la instalación terminó, ejecuta **INICIAR_PAGINA.bat**.

Los entornos virtuales contienen rutas de su instalación: se crean en cada ubicación y equipo. **No copies ni muevas una `.venv` existente para instalar este proyecto en otra carpeta.** El ZIP contiene todo el código, el modelo y los datos de demostración; las bibliotecas de Python se instalan una vez en cada entorno.

## 2. Abrir la página cada vez

En Windows, haz doble clic en **INICIAR_PAGINA.bat**. También puedes abrir `neumonia_app` en VS Code, seleccionar **Terminal → Nueva terminal** y ejecutar:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abre **http://localhost:8501**. La configuración escucha solo en `127.0.0.1`. Mantén abierta la terminal y el computador encendido mientras utilizas la página. Para detenerla, pulsa **Ctrl+C**. Si solo cerraste la pestaña del navegador y el servidor sigue abierto, vuelve a la dirección sin iniciar otra instancia.

La aplicación utiliza CPU. Procesa una imagen por solicitud y no realiza entrenamiento. El entorno virtual se usa directamente; no hace falta activarlo ni reinstalar bibliotecas cada vez.

## 3. Imágenes para la demostración

La carpeta **imagenes_demo/test/** contiene las **624 radiografías del test oficial**: **234 NORMAL** y **390 PNEUMONIA**. Los archivos se contrastaron con las huellas del conjunto que se evaluó durante el proyecto. Se conserva la composición original, incluidos los contenidos repetidos del test; la vista complementaria de 618 contenidos únicos corresponde al mismo conjunto.

Para iniciar la exposición, utiliza:

- `imagenes_demo/00_inicio_exposicion/01_NORMAL.jpeg`
- `imagenes_demo/00_inicio_exposicion/02_NEUMONIA.jpeg`

Son ejemplos seleccionados por su legibilidad y por la coincidencia entre la etiqueta del dataset y la predicción registrada. **No representan una validación clínica ni una nueva evaluación del modelo.** La selección de dos ejemplos no reemplaza la matriz de confusión sobre las 624 imágenes. Conserva la atribución y las fichas de procedencia incluidas en `imagenes_demo/`.

La carpeta `pruebas_portables/` mantiene las 16 imágenes para la comprobación técnica. Esas 16 incluyen errores conocidos y no constituyen una muestra aleatoria para estimar desempeño.

## 4. Las tres pestañas

| Pestaña | Qué permite hacer |
|---|---|
| **Analizar imagen** | Cargar un JPEG/PNG, observar la radiografía y su entrada de 224 × 224, ejecutar el análisis, consultar la clase, el puntaje y el tiempo, y descargar el resultado en JSON. |
| **Resultados del experimento** | Revisar las métricas sobre las 624 imágenes, la matriz de confusión, las políticas de clasificación, las figuras y el registro de errores. |
| **Acerca del modelo** | Comprender la arquitectura, el preprocesamiento, el entrenamiento, la época seleccionada y las conclusiones. |

El resultado JSON permite conservar la salida técnica de una imagen. La aplicación no guarda automáticamente la radiografía ni envía su contenido a servicios externos de IA. El navegador puede mantener el archivo en la sesión mientras la página está abierta.

## 5. Modelo y resultados

- **Arquitectura:** MobileNetV2 con transferencia de aprendizaje desde ImageNet V2; clasificador Dropout(0,2) + Linear(1280,1), seguido de sigmoide para obtener el puntaje.
- **Tamaño:** 2.225.153 parámetros. Entrada de 224 × 224 en tres canales.
- **Datos fijados:** 3.796 imágenes de entrenamiento, 950 de validación y 624 de test.
- **Selección:** época global 15, correspondiente a la época 12 del ajuste fino después de tres épocas de entrenamiento de la cabeza. Modelo y umbral seleccionados con validación.
- **Regla:** PNEUMONIA si el puntaje es igual o superior a **0,8745065331459045**; NORMAL en caso contrario.
- **Test oficial:** 367 verdaderos positivos, 201 verdaderos negativos, 23 falsos negativos y 33 falsos positivos. Exactitud **91,03 %**, sensibilidad **94,10 %** y especificidad **85,90 %**.

El puntaje favorece una clase, pero no expresa la probabilidad de acertar en un paciente. La clasificación NORMAL tampoco descarta otras alteraciones. **El proyecto tiene alcance académico y no sustituye un diagnóstico.** Su rendimiento no se puede generalizar a todas las radiografías de tórax; faltan evaluación externa y verificación de la independencia clínica por paciente.

## 6. Archivos principales

| Archivo o carpeta | Contenido |
|---|---|
| `app.py` | Interfaz de las tres pestañas. |
| `neumonia/inference.py` | Motor común de preprocesamiento e inferencia. |
| `modelo/` | Pesos y metadatos originales del modelo entrenado. |
| `imagenes_demo/` | Test completo y dos ejemplos iniciales para la exposición. |
| `referencia/` | Métricas, predicciones, historial y figuras del experimento. |
| `evidencia/` | Comparación de implementaciones y registro de los 56 errores. |
| `pruebas_portables/` | Referencias e imágenes para comprobar una instalación. |
| `verificar_instalacion.py` | Comprobación técnica CPU local frente a CPU de Kaggle. |
| `requirements.txt` | Versiones de las bibliotecas de la aplicación. |
| `.streamlit/config.toml` | Tema visual y configuración del servidor local. |

## 7. Instalación manual y comprobación opcional

Si prefieres PowerShell en lugar del instalador `.bat`, ejecuta estos comandos, uno a uno, dentro de `neumonia_app`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

La creación del entorno se realiza al instalar en una ubicación nueva. Si `py -3.12` no encuentra Python, instala primero esa versión de 64 bits y abre una terminal nueva. Instalar únicamente la extensión de Python en VS Code no instala el intérprete.

Al cambiar de equipo o de dependencias, puedes comprobar la instalación:

```powershell
.\.venv\Scripts\python.exe verificar_instalacion.py
```

El resultado local esperado es `VERIFICACION_CPU_LOCAL_APROBADA` y se guarda en `evidencia/verificacion_local.json`. No es necesario repetirlo en cada apertura de la aplicación. Su alcance se explica en el apartado técnico siguiente.

## 8. Respaldo técnico conservado

La presentación de la página se ha orientado al producto final. Los registros técnicos conservan sus resultados originales:

- La comparación histórica GPU con precisión mixta frente a CPU float32 procesó **624/624** imágenes, con **624/624** entradas y clases coincidentes. En **5** imágenes, la diferencia de puntaje superó la tolerancia de 0,01; por ello su paridad numérica global sigue **no aprobada**.
- La comprobación posterior del computador de Esteban frente a CPU de Kaggle aprobó sus **16 casos**, con diferencia máxima aproximada de **0,00000209** y tolerancia de 0,0001. Esto comprueba esa instalación y esos casos; no constituye validación externa.
- Los paneles y el registro de **23 falsos negativos y 33 falsos positivos** están disponibles. El registro original mantiene pendiente la revisión humana; no se atribuyen causas clínicas sin evidencia.

La comprobación portable conserva ambas comparaciones por separado. La paridad histórica no se cambia a aprobada al modificar textos o al aprobar una instalación local. Los pesos, el preprocesamiento y el umbral permanecen fijos. La demostración no utiliza el test para reajustar el modelo.

## Equipo y fuentes

**Integrantes:** Juan Esteban Lopez Marin, Juan Camilo Giraldo Jiménez, Juan Pablo Ocampo Santana y Juan Esteban Mora Diaz. **Grupo 2: (Juan)⁴.**

**Profesora:** Isabel Cristina Giraldo Arango. **Materia:** Laboratorio de Inteligencia Artificial aplicado a la Innovación, la Ingeniería y la gestión empresarial. **IA asignada:** Arena. Asistencia de programación y revisión: ChatGPT.

- Dataset de práctica: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Streamlit: https://docs.streamlit.io/get-started/installation/command-line
- PyTorch: https://pytorch.org/get-started/previous-versions/
- Python para Windows: https://www.python.org/downloads/windows/
