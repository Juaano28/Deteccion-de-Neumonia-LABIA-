"""Interfaz de entrega académica del clasificador MobileNetV2.

Conserva la inferencia y los resultados del experimento. La información de uso,
resultados y arquitectura se presenta en tres pestañas; la trazabilidad técnica
queda disponible como información complementaria.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import ImageOps
import streamlit as st

from neumonia.inference import InferenceEngine, input_preview, load_image


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "modelo"
REFERENCE_DIR = ROOT / "referencia"
EVIDENCE_DIR = ROOT / "evidencia"
PREDICTION_KEY = "prediction_fase4"
IMAGE_KEY = "image_digest_fase4"

st.set_page_config(
    page_title="Radiografías · Laboratorio IA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    .stApp { background: #f5f7fa; color: #13283d; }
    .block-container { max-width: 1160px; padding-top: 2.2rem; padding-bottom: 3rem; }
    .hero { background: #102b46; color: #fff; padding: 2rem 2.2rem;
      border-radius: 14px; margin-bottom: 1.1rem; }
    .eyebrow { color: #8edbce; font-size: .76rem; letter-spacing: .12em;
      text-transform: uppercase; font-weight: 700; }
    .hero h1 { color: #fff; font-size: clamp(1.7rem, 4vw, 2.3rem); line-height: 1.2;
      margin: .6rem 0 .7rem; padding: 0; }
    .hero p { color: #d4e0eb; max-width: 760px; margin: 0; font-size: 1rem; }
    .project-tag { display: inline-block; color: #d4f7f0; border: 1px solid #467583;
      border-radius: 20px; padding: .25rem .75rem; margin-top: 1rem; font-size: .78rem; }
    .empty-result { background: #fff; border: 1px dashed #bbcbd8; border-radius: 12px;
      padding: 2rem; margin: 1rem 0; min-height: 160px; }
    .empty-result h3 { color: #25445c; margin-top: 0; }
    .empty-result p { color: #526878; }
    .result-card { background: #fff; border: 1px solid #d5e0e8;
      border-left: 5px solid #128375; padding: 1.2rem 1.4rem; border-radius: 10px;
      margin-top: 1rem; margin-bottom: 1rem; }
    .result-card.pneumonia { border-left-color: #ad6720; }
    .result-card small { font-size: .78rem; color: #526878; text-transform: uppercase;
      letter-spacing: .08em; }
    .result-card h2 { color: #13283d; margin: .35rem 0 0; padding: 0; font-size: 1.75rem; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid #e0e7ee;
      border-radius: 10px; padding: .85rem 1rem; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    .footer-note { border-top: 1px solid #d9e3eb; color: #5b6d7d; font-size: .8rem;
      padding-top: 1rem; margin-top: 2rem; }
    @media (max-width: 700px) { .hero { padding: 1.3rem; }
      .block-container { padding-top: 1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def decimal(value: float, places: int = 4) -> str:
    return f"{float(value):.{places}f}".replace(".", ",")


def file_stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_resource(show_spinner="Cargando el modelo entrenado…")
def get_engine(
    model_dir: str,
    checkpoint_stamp: tuple[int, int],
    metadata_stamp: tuple[int, int],
) -> InferenceEngine:
    # Los sellos cambian la clave de caché si se reemplaza un archivo.
    # El motor vuelve a comprobar las huellas al crear la instancia.
    return InferenceEngine(Path(model_dir), device="cpu")


def read_metadata() -> tuple[dict | None, str | None]:
    try:
        metadata = json.loads((MODEL_DIR / "metadatos_modelo.json").read_text(encoding="utf-8"))
        if not (MODEL_DIR / "global_best.pt").is_file():
            raise FileNotFoundError("Falta modelo/global_best.pt.")
        return metadata, None
    except (OSError, ValueError, TypeError) as exc:
        return None, str(exc)


def clear_prediction(new_digest: str | None) -> None:
    if st.session_state.get(IMAGE_KEY) != new_digest:
        st.session_state[IMAGE_KEY] = new_digest
        st.session_state.pop(PREDICTION_KEY, None)


def export_result(result: dict) -> str:
    """Exporta solo datos técnicos; excluye el nombre y el contenido del archivo."""
    keys = (
        "label", "label_es", "score_pneumonia", "logit", "threshold",
        "inference_ms", "device", "calibrated", "model_sha256",
    )
    payload = {key: result[key] for key in keys if key in result}
    payload["purpose"] = "Prototipo académico; no constituye diagnóstico clínico."
    payload["score_interpretation"] = "Puntaje PNEUMONIA sin calibrar; no probabilidad de acertar."
    payload["positive_rule"] = "score_pneumonia >= threshold"
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


metadata, metadata_error = read_metadata()

st.markdown(
    """<div class="hero">
    <div class="eyebrow">Laboratorio de inteligencia artificial · Proyecto final</div>
    <h1>Clasificación de radiografías de tórax</h1>
    <p>Una aplicación de visión por computador que integra una red neuronal convolucional
    entrenada para clasificar imágenes del conjunto de estudio en NORMAL o NEUMONÍA.</p>
    <span class="project-tag">Grupo 2 · (Juan)⁴ &nbsp; / &nbsp; Entrega académica</span>
    </div>""",
    unsafe_allow_html=True,
)
st.info(
    "Uso académico. Esta aplicación permite explorar un modelo de clasificación de radiografías "
    "y no sustituye un diagnóstico ni una evaluación médica."
)
if metadata_error:
    st.error(f"No se pudo abrir el paquete del modelo: {metadata_error} Extrae el ZIP completo y conserva sus carpetas.")

phase_status_path = EVIDENCE_DIR / "estado_fase4.json"
try:
    phase_status = json.loads(phase_status_path.read_text(encoding="utf-8")) if phase_status_path.is_file() else {}
except (OSError, ValueError):
    phase_status = {}
historical = phase_status.get("parity") or {}
historical_n = historical.get("n_expected")
historical_checked = historical.get("n_checked")

st.write(
    "El proyecto utiliza **MobileNetV2 con aprendizaje por transferencia**, un clasificador binario "
    "y parámetros de entrenamiento definidos antes de evaluar el test. La aplicación reúne la "
    "predicción de imágenes, los resultados del experimento y la explicación de la arquitectura."
)
processed, architecture, input_size = st.columns(3)
processed_value = (
    f"{historical_checked}/{historical_n}"
    if isinstance(historical_checked, int) and isinstance(historical_n, int)
    else "Consultar registro"
)
processed.metric("Imágenes del test procesadas", processed_value)
architecture.metric("Arquitectura", "MobileNetV2")
input_size.metric("Entrada de la red", "224 × 224")
st.caption(
    "El total procesado indica cobertura del test; el desempeño se presenta en «Resultados del experimento». "
    "En «Analizar imagen» puedes cargar una radiografía y en «Acerca del modelo» consultar el entrenamiento "
    "y los criterios que dieron lugar a esta versión."
)

analysis_tab, results_tab, about_tab = st.tabs(
    ["Analizar imagen", "Resultados del experimento", "Acerca del modelo"]
)

with analysis_tab:
    left, right = st.columns([1.1, 1], gap="large")
    image = None
    with left:
        st.subheader("1. Cargar una imagen")
        uploaded = st.file_uploader(
            "Radiografía en JPEG o PNG",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            help="Un archivo de hasta 20 MiB. No admite DICOM, PDF ni imágenes animadas.",
            key="radiograph_upload",
        )
        if uploaded is None:
            clear_prediction(None)
            st.caption(
                "Para la exposición, comienza con las dos imágenes de imagenes_demo/00_inicio_exposicion. "
                "También puedes explorar las 624 radiografías de imagenes_demo/test, organizadas por la etiqueta "
                "NORMAL o PNEUMONIA del dataset. Utiliza imágenes sin información personal visible."
            )
        else:
            raw = uploaded.getvalue()
            digest = hashlib.sha256(raw).hexdigest()
            clear_prediction(digest)
            try:
                image = load_image(raw)
                oriented = ImageOps.exif_transpose(image)
                st.image(oriented.convert("RGB"), caption="Imagen cargada · orientación corregida", width="stretch")
                st.caption(f"Dimensiones: {oriented.width} × {oriented.height} píxeles · {len(raw) / 1024:.0f} KiB")
                with st.expander("Ver la entrada de 224 × 224 píxeles"):
                    st.image(input_preview(image), width=224, caption="Escala de grises y relleno negro, sin recortar la imagen.")
                    st.caption("El motor replica esta imagen en tres canales y aplica la normalización fijada en el entrenamiento.")
            except (OSError, ValueError, RuntimeError) as exc:
                st.session_state.pop(PREDICTION_KEY, None)
                st.error(f"No se puede procesar este archivo: {exc}")

    with right:
        st.subheader("2. Consultar la predicción")
        st.caption("El análisis se ejecuta al pulsar el botón. El umbral permanece fijo.")
        clicked = st.button(
            "Analizar imagen",
            type="primary",
            width="stretch",
            disabled=image is None or metadata is None,
        )
        if clicked and image is not None:
            st.session_state.pop(PREDICTION_KEY, None)
            try:
                engine = get_engine(
                    str(MODEL_DIR),
                    file_stamp(MODEL_DIR / "global_best.pt"),
                    file_stamp(MODEL_DIR / "metadatos_modelo.json"),
                )
                with st.spinner("Procesando la imagen en este equipo…"):
                    st.session_state[PREDICTION_KEY] = engine.predict_image(image)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                st.error(f"No se pudo completar la inferencia: {exc}")

        result = st.session_state.get(PREDICTION_KEY)
        if result is None:
            st.markdown(
                """<div class="empty-result"><h3>Tu resultado aparecerá aquí</h3>
                <p>Carga una imagen y selecciona «Analizar imagen» para obtener la clasificación.</p></div>""",
                unsafe_allow_html=True,
            )
        else:
            positive = result["label"] == "PNEUMONIA"
            label = "NEUMONÍA" if positive else "NORMAL"
            css_class = " pneumonia" if positive else ""
            st.markdown(
                f'<div class="result-card{css_class}"><small>Clase predicha por el modelo</small><h2>{label}</h2></div>',
                unsafe_allow_html=True,
            )
            score = float(result["score_pneumonia"])
            st.metric("Puntaje para PNEUMONIA · sin calibrar", decimal(score))
            st.progress(min(1.0, max(0.0, score)))
            st.caption("Escala de 0 a 1. Un valor alto favorece la clase PNEUMONIA; no expresa la probabilidad de acertar.")
            threshold = float(result["threshold"])
            comparator = "≥" if positive else "<"
            st.write(f"Regla aplicada: **{decimal(score, 6)} {comparator} {decimal(threshold, 6)}** → **{label}**.")
            if not positive:
                st.caption("La clase NORMAL es una salida del clasificador binario; no descarta enfermedad ni otras alteraciones.")
            with st.expander("Detalles de la inferencia"):
                st.write(f"Tiempo de inferencia registrado: **{decimal(result['inference_ms'], 1)} ms**.")
                st.write(f"Dispositivo: **{result['device']}** · Precisión numérica: **float32**.")
                st.write(f"Umbral completo: `{threshold:.16f}`.")
            st.download_button(
                "Descargar resultado (JSON)",
                data=export_result(result),
                file_name="resultado_academico.json",
                mime="application/json",
                width="stretch",
            )
        st.caption(
            "El archivo se procesa en el equipo que ejecuta Streamlit. La aplicación no guarda la imagen "
            "ni envía solicitudes a servicios de IA. El resultado se descarga únicamente si lo solicitas."
        )

with results_tab:
    st.subheader("Desempeño del modelo en el conjunto de test")
    st.write(
        "El experimento se evaluó sobre **624 radiografías: 234 etiquetadas como NORMAL y 390 como PNEUMONIA**. "
        "Las métricas resumen ese conjunto y permanecen como referencia al utilizar la aplicación; "
        "no representan la certeza de una predicción individual."
    )
    metrics_path = REFERENCE_DIR / "evaluacion_test" / "test_metrics.csv"
    try:
        metrics = pd.read_csv(metrics_path)
        official = metrics.loc[
            (metrics["view"] == "official_test") &
            (metrics["threshold_policy"] == "selected_on_validation")
        ].iloc[0]
        for column, (label, field) in zip(st.columns(3), [
            ("Sensibilidad", "sensitivity"), ("Especificidad", "specificity"), ("Exactitud", "accuracy"),
        ]):
            column.metric(label, f"{decimal(100 * official[field], 2)} %")
        st.caption(f"{int(official['n'])} imágenes · Umbral seleccionado en validación: {decimal(official['threshold'], 7)} · AUROC: {decimal(official['auroc'])}.")
        matrix = pd.DataFrame(
            [[int(official["tn"]), int(official["fp"])], [int(official["fn"]), int(official["tp"])]],
            index=["Etiqueta NORMAL", "Etiqueta PNEUMONIA"],
            columns=["Predicción NORMAL", "Predicción PNEUMONIA"],
        )
        st.markdown("**Matriz de confusión del test oficial**")
        st.dataframe(matrix, width="stretch")
        st.caption(
            f"{int(official['fn'])} falsos negativos y {int(official['fp'])} falsos positivos. "
            "La vista de 618 contenidos únicos pertenece al mismo test y no es validación externa."
        )
        with st.expander("Comparar las políticas y las vistas ya registradas"):
            st.write(
                "La tabla muestra cuatro lecturas del **mismo experimento**: dos formas de contar las imágenes "
                "y dos umbrales de decisión registrados. **Test oficial** incluye las 624 imágenes; "
                "**Contenidos únicos** conserva 618 después de agrupar copias de contenido, por lo que es una "
                "vista complementaria del mismo test y no una evaluación independiente."
            )
            st.write(
                "La política **Referencia 0,5** usa el punto de corte convencional. **Elegido en validación** "
                "usa el umbral 0,8745065 fijado con las 950 imágenes de validación: se buscó la mayor "
                "especificidad manteniendo una sensibilidad mínima del 90 % en ese conjunto. "
                "Esta última política es la que aplica la página; el objetivo de validación no garantiza "
                "ese rendimiento en cualquier población futura."
            )
            st.markdown(
                "- **Imágenes:** número de archivos o contenidos incluidos en esa vista.\n"
                "- **Umbral:** puntaje mínimo para asignar PNEUMONIA.\n"
                "- **Sensibilidad:** proporción de imágenes PNEUMONIA identificadas como tales.\n"
                "- **Especificidad:** proporción de imágenes NORMAL clasificadas como normales.\n"
                "- **Exactitud:** proporción de aciertos entre todas las imágenes de la vista.\n"
                "- **AUROC:** capacidad de ordenar ambas clases a lo largo de los umbrales; "
                "no depende de un único punto de corte.\n"
                "- **FN / FP:** número de falsos negativos / falsos positivos."
            )
            st.caption(
                "Las métricas se expresan entre 0 y 1: por ejemplo, 0,9410 equivale aproximadamente al 94,10 %. "
                "FN y FP son recuentos. Compara filas de la misma vista para observar el efecto del umbral."
            )
            shown = metrics[["view", "threshold_policy", "n", "threshold", "sensitivity", "specificity", "accuracy", "auroc", "fn", "fp"]].rename(columns={
                "view": "Vista", "threshold_policy": "Política", "n": "Imágenes", "threshold": "Umbral",
                "sensitivity": "Sensibilidad", "specificity": "Especificidad", "accuracy": "Exactitud",
                "auroc": "AUROC", "fn": "FN", "fp": "FP",
            })
            shown["Vista"] = shown["Vista"].replace({
                "official_test": "Test oficial", "unique_content_view": "Contenidos únicos",
            })
            shown["Política"] = shown["Política"].replace({
                "fixed_0.5": "Referencia 0,5", "selected_on_validation": "Elegido en validación",
            })
            st.dataframe(shown, hide_index=True, width="stretch")
            st.caption("Las cifras son las registradas en el experimento. El umbral de la aplicación permanece fijo y no se reajusta utilizando este test.")
    except (OSError, ValueError, KeyError, IndexError) as exc:
        st.warning(f"No se pudo abrir la tabla de referencia: {exc}")

    figure_map = {
        "Curvas del entrenamiento": "02_curvas_entrenamiento.png",
        "Evaluación del test oficial": "04_test_oficial.png",
        "Fiabilidad de los puntajes": "05_fiabilidad_test.png",
    }
    available = {name: REFERENCE_DIR / "figuras" / filename for name, filename in figure_map.items()
                 if (REFERENCE_DIR / "figuras" / filename).is_file()}
    if available:
        with st.expander("Explorar las figuras originales"):
            choice = st.selectbox("Figura del experimento", list(available))
            st.image(str(available[choice]), width="stretch")

    errors_path = EVIDENCE_DIR / "errores_test.csv"
    if errors_path.is_file():
        with st.expander("Registro de casos para revisar"):
            st.write(
                "Este registro reúne los **56 desacuerdos con las etiquetas del dataset: 23 falsos negativos "
                "y 33 falsos positivos**. Permite documentar los límites del modelo y localizar cada imagen. "
                "Un **FN** es una imagen etiquetada PNEUMONIA que el modelo clasifica como NORMAL; un **FP** "
                "es una imagen etiquetada NORMAL que el modelo clasifica como PNEUMONIA."
            )
            st.markdown(
                "- **Identificación:** `file_id` y `relative_path` permiten localizar el archivo; "
                "`label` y `target` son su etiqueta y código (NORMAL = 0; PNEUMONIA = 1).\n"
                "- **Predicción:** `score_fase3` es el puntaje histórico; `logit_fase3`, la salida previa a la "
                "sigmoide; `predicted_fase3` y `predicted_label_fase3`, la clase resultante.\n"
                "- **Decisión:** `case_type` identifica FN o FP; `threshold` es el umbral utilizado y "
                "`distance_to_threshold`, la distancia absoluta entre puntaje y umbral.\n"
                "- **Archivo y trazabilidad:** `width`, `height` y `bytes` describen tamaño y dimensiones; "
                "las huellas SHA256 comprueban identidad. Los campos de partición y agrupación conservan la auditoría de datos.\n"
                "- **Revisión humana:** `estado_revision_humana`, `observacion_visual`, `limitacion_o_hipotesis`, "
                "`revisor` y `fecha_revision` permiten registrar una revisión posterior."
            )
            st.caption(
                "La revisión humana figura pendiente en el registro original. Una observación visual o una "
                "hipótesis no equivale a confirmar una causa clínica; los identificadores de agrupación "
                "tampoco acreditan por sí solos la identidad de un paciente."
            )
            errors = pd.read_csv(errors_path)
            st.dataframe(errors, hide_index=True, width="stretch")
            st.download_button("Descargar registro de errores", errors_path.read_bytes(), "errores_test.csv", "text/csv")
            st.caption("El registro documenta desacuerdos con las etiquetas del dataset; no aporta por sí mismo una explicación clínica del error.")

    st.warning(
        "Uso académico: estos resultados corresponden al conjunto estudiado y no garantizan el desempeño "
        "en cualquier radiografía de tórax. La independencia clínica por paciente no está verificada "
        "y se requiere evaluación externa para estudiar su generalización."
    )

with about_tab:
    st.subheader("Arquitectura y decisiones de entrenamiento")
    st.write(
        "**MobileNetV2** es una red neuronal convolucional diseñada para extraer características con un "
        "costo computacional contenido. En este proyecto se parte de pesos **ImageNet IMAGENET1K_V2** "
        "y se adapta su clasificador a las dos etiquetas del conjunto de radiografías."
    )
    st.write(
        "Sus bloques residuales invertidos expanden los canales, aplican una convolución **depthwise** "
        "por canal y proyectan la representación a un cuello de botella lineal. Las conexiones residuales "
        "se utilizan cuando las dimensiones lo permiten. Tras el extractor, el promedio global produce "
        "un vector de **1.280 características**; **Dropout(0,2)** actúa durante el entrenamiento y una "
        "capa lineal genera un único **logit**. La sigmoide convierte ese valor en el puntaje PNEUMONIA."
    )
    st.caption(
        "Base técnica: [MobileNetV2, Sandler et al.](https://arxiv.org/abs/1801.04381) · "
        "[Implementación de Torchvision](https://docs.pytorch.org/vision/stable/models/mobilenetv2.html)."
    )
    st.markdown(
        "**Preprocesamiento:** corrección de orientación EXIF, escala de grises, ajuste proporcional bilineal "
        "a 224 × 224, relleno negro centrado, réplica en tres canales y normalización ImageNet. "
        "La inferencia no aplica aumentos aleatorios."
    )
    if metadata is not None:
        selected_epoch = int(metadata["selection"]["global_epoch"])
        selected_stage_epoch = int(metadata["selection"]["epoch"])
        parameter_count = f"{metadata['parameter_count']:,}".replace(",", ".")
        st.write(f"Modelo seleccionado: **época global {selected_epoch}** · Parámetros: **{parameter_count}**.")
        with st.expander("Entrenamiento y época seleccionada"):
            config = metadata["config"]
            st.write(
                f"El entrenamiento se organizó en **{config['head_epochs']} épocas de adaptación del clasificador** "
                f"y **{config['finetune_epochs']} épocas de ajuste fino**. La época global {selected_epoch} "
                f"corresponde a la época {selected_stage_epoch} del ajuste fino. Se eligió por el mayor "
                "**AUROC de validación**, no por el resultado del test. Una época es un recorrido del "
                "conjunto de entrenamiento durante el aprendizaje."
            )
            training_settings = pd.DataFrame([
                ["Datos", "3.796 imágenes de entrenamiento · 950 de validación · 624 de test"],
                ["Pesos iniciales", str(config["weights"])],
                ["Lotes", f"{config['batch_size']} imágenes para entrenar · {config['eval_batch_size']} para evaluar"],
                ["Aprendizaje del clasificador inicial", f"{config['head_lr']:.0e}"],
                ["Aprendizaje en ajuste fino", f"Extractor: {config['finetune_backbone_lr']:.0e} · Clasificador: {config['finetune_head_lr']:.0e}"],
                ["Inicio del ajuste fino", f"Bloque de índice {config['finetune_from']} del extractor y clasificador"],
                ["Regularización", f"Dropout: {decimal(config['dropout'], 1)} · Penalización de pesos: {config['weight_decay']:.0e}"],
                ["Aumentos de entrenamiento", f"Rotación de hasta {config['rotation_degrees']:.0f}° · Traslación de hasta {100 * config['translation_fraction']:.0f} %"],
                ["Balance de clases", "Pérdida ponderada por clase con pesos calculados exclusivamente en entrenamiento"],
                ["Reproducibilidad", f"Semilla {config['seed']} · GPU Tesla T4 en Kaggle · Precisión mixta durante el entrenamiento"],
            ], columns=["Configuración", "Valor registrado"])
            st.dataframe(training_settings, hide_index=True, width="stretch")
            st.write(
                "En el ajuste fino se usan tasas de aprendizaje menores para adaptar las características "
                "preentrenadas de forma gradual. La ponderación de la pérdida compensa la diferencia de "
                "frecuencias de las clases en entrenamiento; no demuestra por sí sola la ausencia de sesgo."
            )
            try:
                history = pd.read_csv(REFERENCE_DIR / "historial_entrenamiento.csv")
                selected = history.loc[history["global_epoch"] == selected_epoch].iloc[0]
                selected_values = pd.DataFrame([
                    ["Pérdida ponderada de entrenamiento", decimal(selected["train_loss_weighted"], 6)],
                    ["Pérdida ponderada de validación", decimal(selected["val_loss_weighted"], 6)],
                    ["Pérdida sin ponderar de validación", decimal(selected["val_loss_unweighted"], 6)],
                    ["AUROC de entrenamiento", decimal(selected["train_auroc"], 6)],
                    ["AUROC de validación · criterio de selección", decimal(selected["val_auroc"], 6)],
                    ["Precisión promedio de validación · AP", decimal(selected["val_average_precision"], 6)],
                    ["Duración de esa época en Kaggle", f"{decimal(selected['elapsed_seconds'], 2)} s"],
                ], columns=[f"Registro de la época global {selected_epoch}", "Valor"])
                st.dataframe(selected_values, hide_index=True, width="stretch")
            except (OSError, ValueError, KeyError, IndexError) as exc:
                st.caption(f"El detalle de la época no está disponible: {exc}")
            st.caption(
                "Las pérdidas miden el error de aprendizaje: la ponderada incorpora el balance de clases y "
                "la no ponderada da el mismo peso a cada imagen. AUROC resume la discriminación y AP resume "
                "la curva precisión–sensibilidad; ninguna equivale al porcentaje de aciertos. "
                "La validación se utilizó para seleccionar esta época y puede ofrecer una estimación optimista. "
                "Las métricas del test se muestran en la pestaña «Resultados del experimento»."
            )
            st.write(
                "**Umbral final: 0,8745065331459045.** Se fijó en validación buscando la mayor especificidad "
                "entre los umbrales con sensibilidad de al menos el 90 %; en un empate se eligió el umbral "
                "mayor. Esa regla definió la decisión antes de evaluar el test."
            )

    st.markdown("**Conclusión del modelo y sus resultados**")
    st.write(
        "La transferencia de aprendizaje permitió integrar una CNN de 2.225.153 parámetros en una "
        "aplicación que ejecuta predicciones individuales en CPU. En el test oficial, el modelo obtuvo "
        "**568 aciertos de 624 imágenes (91,03 % de exactitud)**, con **94,10 % de sensibilidad** y "
        "**85,90 % de especificidad**. Los **23 falsos negativos y 33 falsos positivos** delimitan su "
        "desempeño observado y forman parte de la documentación del proyecto."
    )
    st.write(
        "El resultado es un prototipo académico funcional que conecta preparación de datos, entrenamiento, "
        "evaluación e interfaz de usuario. Su alcance se limita a las dos clases y al conjunto estudiado; "
        "utilizarlo con otras poblaciones, equipos de imagen o contextos clínicos requiere evaluación adicional."
    )
    st.markdown("**Equipo · Grupo 2: (Juan)⁴**")
    st.write("Juan Esteban Lopez Marin · Juan Camilo Giraldo Jiménez · Juan Pablo Ocampo Santana · Juan Esteban Mora Diaz")
    st.write("**Profesora:** Isabel Cristina Giraldo Arango")
    st.write("**Materia:** Laboratorio de Inteligencia Artificial aplicado a la Innovación, la Ingeniería y la gestión empresarial")
    st.caption("IA asignada en clase: Arena. Asistencia en programación y revisión técnica: ChatGPT.")
    with st.expander("Trazabilidad técnica del experimento"):
        st.write(
            "El paquete conserva los pesos, metadatos, resultados e informes originales en las carpetas "
            "`modelo`, `referencia` y `evidencia`. Las comprobaciones de instalación siguen disponibles "
            "mediante `verificar_instalacion.py`."
        )
        if historical.get("n_checked") == 624 and historical.get("class_mismatches") == 0:
            outside = int(historical.get("score_outside_tolerance", 0))
            st.write(
                "En la comparación registrada entre la evaluación en GPU con precisión mixta y la "
                "inferencia CPU float32 se mantuvieron las **624 clasificaciones** y la preparación "
                "de las imágenes. "
                f"**{outside} puntajes** superaron la tolerancia histórica de "
                f"{decimal(float(historical['score_absolute_tolerance']), 2)}. "
                "El registro conserva esa discrepancia; la paridad numérica histórica no se declara aprobada."
            )
        st.caption(
            "La consistencia entre entornos comprueba implementación. No añade casos independientes "
            "ni sustituye una evaluación clínica externa."
        )

st.markdown(
    '<div class="footer-note">Laboratorio IA · Grupo 2 (Juan)⁴ · Desarrollo académico con resultados trazables.</div>',
    unsafe_allow_html=True,
)
