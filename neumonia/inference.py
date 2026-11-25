"""Inferencia reproducible de la fase 3, sin entrenamiento ni calibración.

El módulo utiliza el checkpoint real del experimento 01. No descarga pesos,
no modifica el umbral y no interpreta la sigmoide como certeza diagnóstica.
Las funciones de transformación reproducen CXRTransform(training=False).
"""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import warnings

import numpy as np
import PIL
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
import torch
from torch import nn
from torchvision.models import mobilenet_v2


EXPECTED_CHECKPOINT_SHA256 = "d12e084de1fcf84c5aa57c5f41e73bf8bc5e07730fd2e3a95c232cbddc799c33"
EXPECTED_METADATA_SHA256 = "30583df0c226d7e158c504b1f6fdf80386bb988eb21a5f4928b6ebedffcd63b9"
EXPECTED_CONFIG_SHA256 = "cfbd63bb2139e9c9fb76324dd3d1f84c910b88ffc41596fea6771d05766151f4"
EXPECTED_FROZEN_SHA256 = "b34c6fd669e98231fdcc8f772503197097b846d36ef9772645f43ce0ba82d8c0"
EXPECTED_THRESHOLD = 0.8745065331459045
LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGE_SIDE = 224
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 64_000_000
ALLOWED_FORMATS = {"JPEG", "PNG"}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class InferenceError(ValueError):
    """Error de entrada o de integridad que puede mostrarse en la interfaz."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_image_dimensions(image: Image.Image) -> None:
    if not isinstance(image, Image.Image):
        raise InferenceError("La entrada debe ser una imagen PIL decodificada.")
    if image.width < 1 or image.height < 1:
        raise InferenceError("La imagen no tiene dimensiones válidas.")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise InferenceError("La imagen supera el límite técnico de 64 millones de píxeles.")
    if getattr(image, "n_frames", 1) != 1:
        raise InferenceError("Utiliza una imagen estática de un solo fotograma.")


def load_image(source: bytes | bytearray | memoryview | str | Path) -> Image.Image:
    """Decodifica JPEG/PNG en memoria, conservando EXIF para la transformación.

    Se comprueba el contenido real del archivo, no solamente su extensión.
    Estas verificaciones son técnicas: no determinan si la imagen es una
    radiografía de tórax ni si tiene calidad suficiente para interpretación.
    No se escribe la imagen recibida en disco.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            if not path.is_file():
                raise InferenceError("No se encuentra el archivo de imagen indicado.")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise InferenceError("El archivo supera el límite de 20 MiB.")
            with path.open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise InferenceError("No fue posible leer el archivo de imagen.") from exc
    elif isinstance(source, (bytes, bytearray, memoryview)):
        if len(source) > MAX_FILE_BYTES:
            raise InferenceError("El archivo supera el límite de 20 MiB.")
        raw = bytes(source)
    else:
        raise InferenceError("Utiliza bytes de una imagen o una ruta local.")
    if not raw:
        raise InferenceError("El archivo está vacío.")
    if len(raw) > MAX_FILE_BYTES:
        raise InferenceError("El archivo supera el límite de 20 MiB.")
    try:
        # Mantener la misma política estricta de lectura usada en fase 3.
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise InferenceError("Formato no admitido. Utiliza JPEG o PNG; DICOM requiere otro flujo.")
                _validate_image_dimensions(probe)
                probe.verify()
            with Image.open(BytesIO(raw)) as opened:
                _validate_image_dimensions(opened)
                opened.load()
                decoded = opened.copy()
                decoded.format = opened.format
                return decoded
    except InferenceError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise InferenceError("La imagen supera los límites técnicos de tamaño al decodificarla.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InferenceError("No fue posible decodificar la imagen completa; revisa si está dañada.") from exc


def input_preview(image: Image.Image) -> Image.Image:
    """Lienzo gris 224×224 exacto de evaluación, anterior a la normalización."""
    _validate_image_dimensions(image)
    gray = ImageOps.exif_transpose(image).convert("L")
    scale = min(IMAGE_SIDE / gray.width, IMAGE_SIDE / gray.height)
    width = max(1, min(IMAGE_SIDE, round(gray.width * scale)))
    height = max(1, min(IMAGE_SIDE, round(gray.height * scale)))
    resized = gray.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (IMAGE_SIDE, IMAGE_SIDE), 0)
    canvas.paste(resized, ((IMAGE_SIDE - width) // 2, (IMAGE_SIDE - height) // 2))
    return canvas


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Tensor float32 CHW, equivalente a CXRTransform(False) de fase 3."""
    canvas = input_preview(image)
    values = np.asarray(canvas.convert("RGB"), dtype=np.float32) / np.float32(255)
    normalized = (values - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))


def _load_verified_files(model_dir: Path) -> tuple[dict, dict]:
    checkpoint_path = model_dir / "global_best.pt"
    metadata_path = model_dir / "metadatos_modelo.json"
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise InferenceError(
            "Faltan global_best.pt o metadatos_modelo.json en la carpeta del modelo. "
            "Conserva la estructura completa del paquete."
        )
    if file_sha256(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise InferenceError("El checkpoint no coincide con el modelo evaluado en la fase 3.")
    if file_sha256(metadata_path) != EXPECTED_METADATA_SHA256:
        raise InferenceError("Los metadatos cambiaron. Restaura el archivo original de la fase 3.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # weights_only evita reconstruir objetos arbitrarios almacenados con pickle.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = metadata["config"]
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    checks = [
        checkpoint.get("format") == "cxr_mobilenetv2_training_v1",
        checkpoint.get("checkpoint_type") == "selected_model",
        checkpoint.get("test_mode") is False,
        metadata.get("test_mode") is False,
        metadata.get("calibrated") is False,
        config_hash == EXPECTED_CONFIG_SHA256,
        checkpoint.get("config_hash") == config_hash == metadata.get("config_hash"),
        checkpoint.get("config") == config,
        checkpoint.get("manifest_hash") == metadata.get("frozen_package_sha256") == EXPECTED_FROZEN_SHA256,
        checkpoint.get("label_map") == metadata.get("label_map") == LABEL_MAP,
        metadata.get("selected_checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256,
        metadata.get("threshold") == EXPECTED_THRESHOLD,
        metadata.get("positive_rule") == "score >= threshold",
        metadata.get("input_shape") == [3, IMAGE_SIDE, IMAGE_SIDE],
        metadata.get("preprocessing", {}).get("evaluation_augmentation") is False,
        config.get("image_size") == IMAGE_SIDE,
        config.get("dropout") == 0.2,
        checkpoint.get("best_validation") == metadata.get("selection"),
    ]
    if not all(checks):
        raise InferenceError("Checkpoint, configuración y metadatos no representan el mismo experimento.")
    return metadata, checkpoint


class InferenceEngine:
    """Carga una sola vez el modelo real y predice en float32, sin aumentos.

    CPU es el dispositivo predeterminado para una interfaz local reproducible.
    El bloqueo permite compartir esta instancia mediante st.cache_resource.
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        self.model_dir = Path(model_dir).resolve()
        self.metadata, checkpoint = _load_verified_files(self.model_dir)
        if PIL.__version__ != self.metadata["environment"]["pillow"]:
            raise InferenceError(
                f"Esta versión conserva Pillow {self.metadata['environment']['pillow']} del entrenamiento; "
                f"se encontró {PIL.__version__}. Instala las dependencias de requirements.txt y reinicia "
                "el entorno. Pillow 13 cambió el redimensionamiento de algunas proporciones de imagen."
            )
        device_name = str(device)
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        chosen = torch.device(device_name)
        if chosen.type not in {"cpu", "cuda"}:
            raise InferenceError("Esta versión comprueba inferencia en CPU y CUDA; utiliza uno de esos dispositivos.")
        if chosen.type == "cuda" and not torch.cuda.is_available():
            raise InferenceError("Se solicitó CUDA, pero no hay una GPU CUDA disponible.")
        if chosen.type == "cuda" and chosen.index is not None and chosen.index >= torch.cuda.device_count():
            raise InferenceError("El índice de GPU solicitado no existe en esta sesión.")
        if chosen.type == "cpu":
            desired_threads = max(1, min(4, os.cpu_count() or 1))
            if torch.get_num_threads() > desired_threads:
                torch.set_num_threads(desired_threads)
        self.device = chosen
        self._threshold = float(self.metadata["threshold"])
        self.model_sha256 = EXPECTED_CHECKPOINT_SHA256
        self._lock = threading.RLock()
        # La arquitectura se inicializa sin descargar ImageNet y TODOS sus
        # parámetros/buffers se sustituyen por el estado entrenado verificado.
        self.model = mobilenet_v2(weights=None)
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(self.model.last_channel, 1))
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self.model.requires_grad_(False)
        if sum(parameter.numel() for parameter in self.model.parameters()) != self.metadata["parameter_count"]:
            raise InferenceError("La arquitectura reconstruida no tiene el número de parámetros esperado.")

    @property
    def threshold(self) -> float:
        return self._threshold

    def predict_image(self, image: Image.Image) -> dict:
        """Clasifica una imagen; el puntaje no es una probabilidad de acertar."""
        started = time.perf_counter()
        tensor = preprocess_image(image).unsqueeze(0).to(device=self.device, dtype=torch.float32)
        with self._lock, torch.inference_mode():
            self.model.eval()
            # Desactivar AMP incluso si el llamador está dentro de autocast.
            autocast_off = torch.autocast(device_type=self.device.type, enabled=False)
            with autocast_off:
                output = self.model(tensor).reshape(-1)
                if output.numel() != 1 or not torch.isfinite(output).all():
                    raise InferenceError("El modelo produjo una salida no válida; no se presenta una clasificación.")
                logit = float(output[0].item())
                score = float(torch.sigmoid(output[0]).item())
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        if not (math.isfinite(score) and 0.0 <= score <= 1.0):
            raise InferenceError("El puntaje calculado no es válido.")
        positive = score >= self.threshold
        return {
            "label": "PNEUMONIA" if positive else "NORMAL",
            "label_es": "Neumonía" if positive else "Normal",
            "score_pneumonia": score,
            "logit": logit,
            "threshold": self.threshold,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
            "device": str(self.device),
            "precision": "float32_sin_amp",
            "calibrated": False,
            "model_sha256": self.model_sha256,
            "score_interpretation": "Puntaje del modelo sin calibrar; no es probabilidad de acertar.",
            "scope": "Prototipo académico binario; no diagnóstico clínico.",
        }

    def predict_bytes(self, raw: bytes) -> dict:
        return self.predict_image(load_image(raw))
