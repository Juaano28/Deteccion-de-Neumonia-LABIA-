"""Extractos literales del cuaderno de fase 3, usados solo para comprobar paridad.
CXRTransform(False) no aplica aumentos. No contiene entrenamiento.
"""
import json
import math
import numpy as np
import torch
from PIL import Image, ImageOps
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def default_training_config():
    return {
        "seed": 42, "weights": "IMAGENET1K_V2", "image_size": 224,
        "dropout": 0.2, "batch_size": 32, "eval_batch_size": 64, "num_workers": 2,
        "rotation_degrees": 5.0, "translation_fraction": 0.02,
        "head_epochs": 3, "finetune_epochs": 12, "finetune_from": 14,
        "head_lr": 1e-3, "finetune_backbone_lr": 1e-5, "finetune_head_lr": 1e-4,
        "weight_decay": 1e-4, "patience": 4, "amp": True, "ece_bins": 10,
        "loss_balance": "per_example_N_over_2Nc", "deterministic_algorithms_warn_only": True,
    }

def _config(config=None):
    result = default_training_config()
    if config:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f"Opciones desconocidas: {sorted(unknown)}")
        result.update(config)
    if result["weights"] != "IMAGENET1K_V2" or result["image_size"] != 224:
        raise ValueError("Esta receta usa IMAGENET1K_V2 e imágenes 224x224.")
    for key in ("batch_size", "eval_batch_size", "head_epochs", "finetune_epochs", "patience", "ece_bins"):
        if not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"{key} debe ser un entero positivo.")
    if not isinstance(result["num_workers"], int) or result["num_workers"] < 0:
        raise ValueError("num_workers debe ser un entero no negativo.")
    if not 0 <= result["dropout"] < 1 or not 0 <= result["translation_fraction"] <= 0.05:
        raise ValueError("Dropout o traslación fuera de los límites de la receta.")
    if not 0 <= result["rotation_degrees"] <= 10:
        raise ValueError("La rotación debe estar entre 0 y 10 grados.")
    for key in ("head_lr", "finetune_backbone_lr", "finetune_head_lr"):
        if not math.isfinite(float(result[key])) or result[key] <= 0:
            raise ValueError(f"{key} debe ser positivo y finito.")
    if result["weight_decay"] < 0 or result["loss_balance"] != "per_example_N_over_2Nc":
        raise ValueError("Configuración de pérdida/regularización no admitida.")
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result

class CXRTransform:
    """Evaluación idéntica a fase 2A; aumentos moderados exclusivamente en train.

    En train, la rotación amplía el lienzo para no cortar esquinas. Se reserva
    margen para trasladar sin recortar. No se aplican flips ni crops aleatorios.
    """
    def __init__(self, training=False, config=None):
        self.training = bool(training)
        self.config = _config(config)

    def __call__(self, image):
        gray = ImageOps.exif_transpose(image).convert("L")
        side = self.config["image_size"]
        margin, dx, dy = 0, 0, 0
        if self.training:
            degrees = self.config["rotation_degrees"]
            angle = float(torch.empty(1).uniform_(-degrees, degrees).item()) if degrees else 0.0
            if angle:
                gray = gray.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=0)
            translation = self.config["translation_fraction"]
            margin = int(math.ceil(side * translation)) if translation else 0
            if margin:
                dx = int(round(float(torch.empty(1).uniform_(-side*translation, side*translation).item())))
                dy = int(round(float(torch.empty(1).uniform_(-side*translation, side*translation).item())))
        available = side - 2*margin
        scale = min(available / gray.width, available / gray.height)
        width = max(1, min(available, round(gray.width * scale)))
        height = max(1, min(available, round(gray.height * scale)))
        resized = gray.resize((width, height), Image.Resampling.BILINEAR)
        canvas = Image.new("L", (side, side), 0)
        left, top = (side-width)//2 + dx, (side-height)//2 + dy
        if left < 0 or top < 0 or left+width > side or top+height > side:
            raise RuntimeError("Un aumento excedió el margen; no se permite recortar la entrada.")
        canvas.paste(resized, (left, top))
        values = np.asarray(canvas.convert("RGB"), dtype=np.float32) / np.float32(255)
        normalized = (values-MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
