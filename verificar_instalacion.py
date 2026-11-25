"""Comprueba CPU local frente a CPU Kaggle sobre 16 casos portables.

Versión 4.1: el resultado local y la comparación histórica GPU/AMP frente a
CPU/FP32 se informan por separado. Aprobar esta regresión local no aprueba la
paridad histórica ni constituye validación clínica. No cambia tolerancias,
pesos, etiquetas ni umbral.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import platform
import sys

import numpy as np
import PIL
import torch
import torchvision

from neumonia.inference import InferenceEngine, load_image, preprocess_image
from neumonia.phase4 import image_path, sha256_file, verify_preservation


def main():
    root = Path(__file__).resolve().parent
    descriptor = root / "pruebas_portables/casos.json"
    if not descriptor.is_file():
        print("PENDIENTE: primero ejecuta el cuaderno 05 en Kaggle y descarga su ZIP de salida.\n"
              "Ese ZIP incorpora 16 imágenes y resultados para comprobar esta instalación.\n"
              "El paquete inicial contiene el modelo, pero aún no esas comprobaciones reales.")
        return 2
    verify_preservation(root)
    cases = json.loads(descriptor.read_text(encoding="utf-8"))
    if cases.get("n") != 16 or len(cases.get("cases", [])) != 16:
        raise ValueError("El paquete de comprobación debe contener 16 casos.")
    if cases.get("score_absolute_tolerance") != 0.0001:
        raise ValueError("La tolerancia de la comprobación fue modificada.")
    historical_passed = cases.get("phase4_kaggle_parity_passed")
    if not isinstance(historical_passed, bool):
        raise ValueError("El descriptor no conserva un estado histórico booleano válido.")
    engine = InferenceEngine(root / "modelo", device="cpu")
    if (cases.get("model_sha256") != engine.model_sha256
            or cases.get("threshold") != engine.threshold
            or len({item["file_id"] for item in cases["cases"]}) != 16):
        raise ValueError("Los casos portables no corresponden al modelo, umbral o identificadores de esta versión.")
    records = []
    for case in cases["cases"]:
        record = {"file_id": case["file_id"], "image_ok": False, "tensor_ok": False,
                  "score_ok": False, "class_ok": False, "abs_score_difference": None, "error": ""}
        try:
            path = image_path(descriptor.parent, case["path"])
            record["image_ok"] = sha256_file(path) == case["image_sha256"]
            if not record["image_ok"]:
                raise ValueError("La imagen no coincide con el archivo original exportado.")
            image = load_image(path)
            tensor = preprocess_image(image)
            record["tensor_ok"] = hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest() == case["tensor_sha256"]
            result = engine.predict_image(image)
            difference = abs(result["score_pneumonia"] - case["score_cpu_kaggle"])
            record.update(abs_score_difference=difference, score_ok=difference <= 0.0001,
                          class_ok=int(result["label"] == "PNEUMONIA") == case["predicted_cpu_kaggle"])
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        ok = all(record[x] for x in ("image_ok", "tensor_ok", "score_ok", "class_ok"))
        print(f"{record['file_id']}: {'OK' if ok else 'REVISAR'}")
    local_passed = bool(len(records) == 16 and all(
        all(row[key] for key in ("image_ok", "tensor_ok", "score_ok", "class_ok")) for row in records))
    full_migration_passed = bool(local_passed and historical_passed)
    differences = [row["abs_score_difference"] for row in records if row["abs_score_difference"] is not None]
    summary = {"schema_version": "4.1",
               "status": "VERIFICACION_CPU_LOCAL_APROBADA" if local_passed else "VERIFICACION_CPU_LOCAL_INCOMPLETA",
               "passed": local_passed, "local_cpu_parity_passed": local_passed,
               "n_cases": len(records), "scope": "local_cpu_vs_kaggle_cpu_on_16_portable_cases",
               "verified_at_utc": datetime.now(timezone.utc).isoformat(),
               "score_absolute_tolerance": 0.0001,
               "max_abs_score_difference": max(differences) if differences else None,
               "historical_parity_passed": historical_passed,
               "kaggle_parity_passed": historical_passed,
               "full_migration_parity_passed": full_migration_passed,
               "model_sha256": sha256_file(root / "modelo/global_best.pt"),
               "metadata_sha256": sha256_file(root / "modelo/metadatos_modelo.json"),
               "portable_descriptor_sha256": sha256_file(descriptor),
               "threshold": engine.threshold,
               "interpretation": "passed describe únicamente la regresión CPU local frente a CPU Kaggle sobre 16 casos. "
                                 "La paridad histórica GPU/AMP frente a CPU/FP32 conserva su resultado original; "
                                 "full_migration_parity_passed exige ambas comprobaciones. Ninguna es validación clínica.",
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "torchvision": torchvision.__version__, "pillow": PIL.__version__,
                               "numpy": np.__version__, "platform": platform.platform(),
                               "machine": platform.machine(), "processor": platform.processor(),
                               "logical_cpu_count": os.cpu_count(), "torch_cpu_threads": torch.get_num_threads(),
                               "device": str(engine.device), "precision": "float32_without_amp"},
               "reference_environment_cpu_kaggle": cases.get("environment_cpu_kaggle", {}),
               "cases": records}
    destination = root / "evidencia/verificacion_local.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary["status"])
    print("Comparación local: CPU de este equipo frente a CPU de Kaggle, sobre 16 casos portables.")
    print("Paridad histórica GPU/AMP frente a CPU/FP32:", "APROBADA" if historical_passed else "NO APROBADA (se conserva el resultado original).")
    print("Paridad agregada de migración completa:", "APROBADA" if full_migration_passed else "NO APROBADA.")
    print(f"Registro: {destination}")
    if local_passed:
        print("Puedes probar la interfaz académica con esta instalación CPU. "
              "Esta aprobación local no cambia el estado histórico ni acredita desempeño clínico.")
    else:
        print("Conserva este registro y compártelo. No cambies pesos, umbral ni tolerancias para aprobar.")
    return 0 if local_passed else 1


if __name__ == "__main__":
    sys.exit(main())
