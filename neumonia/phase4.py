"""Fase 4: conservar el experimento, describir errores y verificar inferencia.

No entrena, no calibra y no selecciona umbrales. Los puntajes históricos se
conservan; cualquier inferencia nueva se registra como comprobación técnica.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import time
from datetime import datetime, timezone
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

EXPECTED_INDEX_SHA256 = "4a35d53ba1a59ebf42149d64244f9cbc366b912b4bce9d15d8c6a689cdfe34f7"
EXPECTED_MODEL_SHA256 = "d12e084de1fcf84c5aa57c5f41e73bf8bc5e07730fd2e3a95c232cbddc799c33"
EXPECTED_METADATA_SHA256 = "30583df0c226d7e158c504b1f6fdf80386bb988eb21a5f4928b6ebedffcd63b9"
DEFAULT_SCORE_TOLERANCE = 0.01
MAX_SOURCE_BYTES = 100 * 1024 ** 2


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value):
    value = str(value)
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or "\\" in value or "\x00" in value
            or ":" in value or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError(f"Ruta relativa no permitida: {value!r}")
    return path


def image_path(root, relative):
    root = Path(root).resolve()
    path = (root / str(_safe_relative(relative))).resolve()
    if not path.is_relative_to(root):
        raise ValueError("La ruta resuelta sale de la carpeta de imágenes.")
    return path


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _read_phase3(source):
    """Acepta el ZIP o su carpeta, incluso con un prefijo de carpeta en el ZIP."""
    if "://" in str(source):
        raise ValueError("Indica una ruta de /kaggle/input, no un enlace web.")
    source = Path(source)
    if source.name == "metadatos_modelo.json":
        source = source.parent
    raw = {}
    if source.is_dir():
        index_path = image_path(source, "indice_sha256.json")
        if not index_path.is_file() or index_path.stat().st_size > 128 * 1024:
            raise ValueError("La carpeta no contiene el índice original de fase 3.")
        index_bytes = index_path.read_bytes()
        if hashlib.sha256(index_bytes).hexdigest() != EXPECTED_INDEX_SHA256:
            raise ValueError("El índice no corresponde al experimento 01 aprobado.")
        index = json.loads(index_bytes)
        total = len(index_bytes)
        for name, expected in index.items():
            path = image_path(source, name)
            total += path.stat().st_size
            if total > MAX_SOURCE_BYTES:
                raise ValueError("La entrada excede el tamaño previsto.")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"No coincide la huella del archivo original: {name}")
            raw[name] = data
        raw["indice_sha256.json"] = index_bytes
    elif source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            if len(infos) > 500 or sum(x.file_size for x in infos) > MAX_SOURCE_BYTES:
                raise ValueError("El ZIP excede los límites del paquete de fase 3.")
            files = {}
            for info in infos:
                if info.is_dir():
                    continue
                _safe_relative(info.filename)
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("El ZIP contiene un enlace simbólico.")
                if info.filename in files:
                    raise ValueError("El ZIP contiene nombres repetidos.")
                files[info.filename] = info
            matches = [name for name in files if PurePosixPath(name).name == "indice_sha256.json"
                       and files[name].file_size < 128 * 1024
                       and hashlib.sha256(archive.read(name)).hexdigest() == EXPECTED_INDEX_SHA256]
            if len(matches) != 1:
                raise ValueError("El ZIP debe contener un único índice del experimento 01.")
            index_name = matches[0]
            prefix = index_name[:-len("indice_sha256.json")]
            index_bytes = archive.read(index_name)
            index = json.loads(index_bytes)
            for name, expected in index.items():
                _safe_relative(name)
                data = archive.read(prefix + name)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"No coincide la huella del archivo original: {name}")
                raw[name] = data
            raw["indice_sha256.json"] = index_bytes
    else:
        raise FileNotFoundError(f"No existe un ZIP o una carpeta legible: {source}")
    if hashlib.sha256(raw["global_best.pt"]).hexdigest() != EXPECTED_MODEL_SHA256:
        raise ValueError("El modelo no es el checkpoint aprobado.")
    if hashlib.sha256(raw["metadatos_modelo.json"]).hexdigest() != EXPECTED_METADATA_SHA256:
        raise ValueError("Los metadatos no son los aprobados.")
    return raw


def discover_phase3(search_root="/kaggle/input", override=None):
    if override is not None:
        _read_phase3(override)
        value = Path(override)
        return value.parent if value.name == "metadatos_modelo.json" else value
    candidates, errors = set(), []
    root = Path(search_root)
    for current, folders, files in os.walk(root, followlinks=False):
        path = Path(current)
        folders[:] = sorted(x for x in folders if not x.startswith(".") and x != "__MACOSX")
        if len(path.parts) - len(root.parts) > 10:
            folders[:] = []
            continue
        if "metadatos_modelo.json" in files and "indice_sha256.json" in files:
            candidates.add(path)
        for name in files:
            candidate = path / name
            if name.lower().endswith(".zip") and candidate.stat().st_size <= MAX_SOURCE_BYTES:
                candidates.add(candidate)
    for candidate in sorted(candidates, key=lambda p: (len(p.parts), str(p))):
        try:
            _read_phase3(candidate)
            return candidate
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            errors.append({"ruta": str(candidate), "motivo": str(exc)})
    raise FileNotFoundError(
        "Añade como Input los resultados completos de fase 3. Se acepta "
        "fase3_mobilenetv2_experimento01_resultados.zip o su carpeta descomprimida. "
        "El ZIP de fase 2B solo no contiene el modelo. No necesitas pegar un enlace web. "
        + json.dumps(errors[:5], ensure_ascii=False))


def preserve_phase3(source, project_root):
    project_root = Path(project_root)
    if project_root.resolve().is_relative_to(Path("/kaggle/input")):
        raise ValueError("Las salidas se escriben fuera de /kaggle/input.")
    raw = _read_phase3(source)
    files = []
    for name, data in raw.items():
        destination = (Path("modelo") / name if name in {"global_best.pt", "metadatos_modelo.json"}
                       else Path("referencia") / name)
        target = project_root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append({"archivo_original": name, "archivo_conservado": destination.as_posix(),
                      "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    preservation = {"experiment": "MobileNetV2_experimento01", "copied_at_utc": _utc_now(),
                    "source_type": "directory" if Path(source).is_dir() else "zip",
                    "source_path": str(source), "original_files_preserved": len(files),
                    "bytes_unchanged": True, "files": files}
    _write_json(project_root / "evidencia/preservacion_experimento.json", preservation)
    return preservation


def verify_preservation(project_root):
    root = Path(project_root)
    manifest = json.loads((root / "evidencia/preservacion_experimento.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = image_path(root, item["archivo_conservado"])
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Cambió un archivo de referencia: {item['archivo_original']}")
    return {"original_files_verified": len(manifest["files"]), "all_ok": True}


def load_reference(project_root):
    root = Path(project_root)
    verify_preservation(root)
    meta = json.loads((root / "modelo/metadatos_modelo.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(root / "referencia/datos_congelados/test_oficial.csv", keep_default_na=False)
    predictions = pd.read_csv(root / "referencia/evaluacion_test/predictions_test_official.csv")
    if len(manifest) != 624 or set(manifest.file_id) != set(predictions.file_id):
        raise ValueError("No coincide el test oficial de 624 imágenes.")
    table = manifest.merge(predictions.rename(columns={"target": "target_prediction", "logit": "logit_fase3",
                            "score_pneumonia": "score_fase3"}), on="file_id", validate="one_to_one")
    if not table.target.eq(table.target_prediction).all():
        raise ValueError("Las etiquetas del manifiesto y las predicciones no coinciden.")
    if not np.isfinite(table.score_fase3).all() or not table.score_fase3.between(0, 1).all():
        raise ValueError("Los puntajes históricos no son válidos.")
    table = table.drop(columns="target_prediction")
    table["predicted_fase3"] = (table.score_fase3 >= float(meta["threshold"])).astype(int)
    table["predicted_label_fase3"] = table.predicted_fase3.map({0: "NORMAL", 1: "PNEUMONIA"})
    table["case_type"] = ["TP" if y == 1 and p == 1 else "TN" if y == 0 and p == 0
                          else "FN" if y == 1 else "FP"
                          for y, p in zip(table.target, table.predicted_fase3)]
    table["threshold"] = float(meta["threshold"])
    table["distance_to_threshold"] = (table.score_fase3 - table.threshold).abs()
    observed = table.case_type.value_counts().to_dict()
    if observed != {"TP": 367, "TN": 201, "FP": 33, "FN": 23}:
        raise ValueError(f"La matriz de confusión de referencia cambió: {observed}")
    return {"table": table, "metadata": meta,
            "metrics": pd.read_csv(root / "referencia/evaluacion_test/test_metrics.csv")}


def build_error_catalog(project_root):
    table = load_reference(project_root)["table"]
    errors = table.loc[table.case_type.isin(["FN", "FP"])].copy()
    errors = errors.sort_values(["case_type", "score_fase3", "file_id"]).reset_index(drop=True)
    errors["estado_revision_humana"] = "pendiente"
    errors["observacion_visual"] = ""
    errors["limitacion_o_hipotesis"] = ""
    errors["revisor"] = ""
    errors["fecha_revision"] = ""
    destination = Path(project_root) / "evidencia"
    destination.mkdir(parents=True, exist_ok=True)
    catalog_path = destination / "errores_test.csv"
    if catalog_path.is_file():
        previous = pd.read_csv(catalog_path, keep_default_na=False)
        if ("file_id" not in previous or previous.file_id.duplicated().any()
                or set(previous.file_id) != set(errors.file_id)):
            raise ValueError("El catálogo existente tiene identificadores distintos. "
                             "Consérvalo por separado antes de regenerar; no se sobrescriben revisiones ajenas.")
        previous = previous.set_index("file_id")
        for column in ("estado_revision_humana", "observacion_visual", "limitacion_o_hipotesis", "revisor", "fecha_revision"):
            if column in previous.columns:
                errors[column] = errors.file_id.map(previous[column])
    errors.to_csv(destination / "errores_test.csv", index=False, encoding="utf-8-sig")
    table.to_csv(destination / "predicciones_referencia_con_rutas.csv", index=False, encoding="utf-8-sig")
    return errors


def locate_images(table, search_root="/kaggle/input", override=None):
    candidates = set()
    if override is not None:
        if "://" in str(override):
            raise ValueError("DATA_ROOT_OVERRIDE debe ser una ruta local, no una URL.")
        candidates.add(Path(override))
    else:
        root = Path(search_root)
        for current, folders, _ in os.walk(root, followlinks=False):
            path = Path(current)
            folders[:] = sorted(x for x in folders if not x.startswith(".") and x != "__MACOSX")
            if len(path.parts) - len(root.parts) > 10:
                folders[:] = []
                continue
            if all((path / "test" / label).is_dir() for label in ("NORMAL", "PNEUMONIA")):
                candidates.add(path)
    records = []
    for candidate in sorted(candidates, key=lambda p: (len(p.parts), str(p))):
        count = 0
        for row in table.itertuples():
            try:
                path = image_path(candidate, row.relative_path)
                count += path.is_file() and path.stat().st_size == int(row.bytes)
            except OSError:
                pass
        records.append({"root": str(candidate), "expected": len(table), "matching_size": count})
    result = pd.DataFrame(records, columns=["root", "expected", "matching_size"])
    valid = result.loc[result.matching_size.eq(len(table))]
    if valid.empty:
        raise FileNotFoundError("Añade Chest X-Ray Images (Pneumonia) a Input. "
                                "No se encontró una copia completa del test original.")
    return Path(valid.iloc[0].root), result


def verify_images(table, data_root, output_dir):
    start, records = time.perf_counter(), []
    for number, row in enumerate(table.itertuples(), 1):
        record = {"file_id": row.file_id, "relative_path": row.relative_path,
                  "expected_sha256": row.byte_sha256, "observed_sha256": "", "status": "ERROR", "error": ""}
        try:
            digest = sha256_file(image_path(data_root, row.relative_path))
            record.update(observed_sha256=digest, status="OK" if digest == row.byte_sha256 else "DIFERENTE")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if number % 100 == 0 or number == len(table):
            print(f"Integridad de imágenes: {number}/{len(table)} | {time.perf_counter()-start:.1f} s", flush=True)
    result = pd.DataFrame(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "integridad_imagenes_fase4.csv", index=False, encoding="utf-8-sig")
    return result


def make_error_panels(errors, data_root, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .inference import load_image, input_preview

    output_dir = Path(output_dir)
    panels = output_dir / "paneles_errores"
    panels.mkdir(parents=True, exist_ok=True)
    output = []
    for case in ("FN", "FP"):
        subset = errors.loc[errors.case_type.eq(case)].copy()
        # FN: menor score primero; FP: mayor score primero. Se incluyen TODOS.
        subset = subset.sort_values(["score_fase3", "file_id"], ascending=[case == "FN", True])
        for start in range(0, len(subset), 8):
            page = subset.iloc[start:start+8]
            nrows = (len(page) + 1) // 2
            ncols = 2 if len(page) == 1 else 4
            height = 3 * nrows + 0.65
            fig, axes = plt.subplots(nrows, ncols, figsize=(7 if ncols == 2 else 14, height), squeeze=False)
            for axis in axes.flat:
                axis.axis("off")
            for index, row in enumerate(page.itertuples()):
                image = load_image(image_path(data_root, row.relative_path))
                r, c = index // 2, (index % 2) * 2
                axes[r, c].imshow(ImageOps.exif_transpose(image).convert("L"), cmap="gray", vmin=0, vmax=255)
                axes[r, c].set_title(f"{row.file_id} · {case}\nEtiqueta: {row.label}\n"
                                    f"Predicción: {row.predicted_label_fase3}", fontsize=9)
                axes[r, c+1].imshow(input_preview(image), cmap="gray", vmin=0, vmax=255)
                axes[r, c+1].set_title(f"Entrada 224 × 224\nScore histórico: {row.score_fase3:.6f}\n"
                                      f"Umbral: {row.threshold:.6f}", fontsize=9)
            fig.suptitle(f"Errores de fase 3 · {case} · casos {start+1}–{start+len(page)} de {len(subset)}\n"
                         "Descripción técnica; la causa clínica no ha sido determinada", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / height))
            filename = panels / f"{case}_{start//8+1:02d}.jpg"
            fig.savefig(filename, dpi=140, pil_kwargs={"quality": 90})
            plt.close(fig)
            output.append(filename)
    if output:
        with Image.open(output[0]) as image:
            image.thumbnail((800, 700))
            image.convert("RGB").save(output_dir / "vista_previa_errores.jpg", quality=55)
    return output


def check_parity(table, engine, data_root, output_dir, tolerance=DEFAULT_SCORE_TOLERANCE):
    import torch
    import torchvision
    import PIL
    from .inference import load_image, preprocess_image
    from referencia_codigo.preprocesamiento_fase3 import CXRTransform

    if float(tolerance) != DEFAULT_SCORE_TOLERANCE:
        raise ValueError("Este protocolo fija tolerancia absoluta 0,01 antes de comparar. "
                         "No se cambia para aprobar una ejecución.")
    if engine.device.type != "cpu":
        raise ValueError("La comprobación de traslado al frontend debe ejecutarse con device='cpu' y FP32.")
    reference_transform = CXRTransform(training=False)
    rows, start = [], time.perf_counter()
    for number, row in enumerate(table.itertuples(), 1):
        record = {"file_id": row.file_id, "relative_path": row.relative_path,
                  "target": int(row.target), "score_fase3": float(row.score_fase3),
                  "logit_fase3": float(row.logit_fase3), "predicted_fase3": int(row.predicted_fase3),
                  "threshold": float(row.threshold), "preprocessing_equal": False,
                  "score_fase4": None, "logit_fase4": None, "abs_score_difference": None,
                  "predicted_fase4": None, "class_equal": False,
                  "score_within_tolerance": False, "repeat_abs_difference": None,
                  "inference_ms": None, "error": ""}
        try:
            image = load_image(image_path(data_root, row.relative_path))
            old_tensor, new_tensor = reference_transform(image), preprocess_image(image)
            record["preprocessing_equal"] = bool(torch.equal(old_tensor, new_tensor))
            prediction = engine.predict_image(image)
            difference = abs(prediction["score_pneumonia"] - row.score_fase3)
            label = int(prediction["label"] == "PNEUMONIA")
            record.update(score_fase4=prediction["score_pneumonia"], logit_fase4=prediction["logit"],
                          abs_score_difference=difference, predicted_fase4=label,
                          class_equal=label == row.predicted_fase3,
                          score_within_tolerance=difference <= tolerance,
                          inference_ms=prediction["inference_ms"])
            if number <= 8:
                repeated = engine.predict_image(image)
                record["repeat_abs_difference"] = abs(repeated["score_pneumonia"] - prediction["score_pneumonia"])
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(record)
        if number % 50 == 0 or number == len(table):
            elapsed = time.perf_counter()-start
            print(f"Paridad CPU FP32: {number}/{len(table)} | {elapsed:.1f} s | "
                  f"restante orientativo {(len(table)-number)*elapsed/number/60:.1f} min", flush=True)
    frame = pd.DataFrame(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "comparacion_paridad.csv", index=False, encoding="utf-8-sig")
    valid = frame.loc[frame.error.eq("")]
    diff = valid.abs_score_difference.astype(float)
    repetitions = valid.repeat_abs_difference.dropna().astype(float)
    summary = {"purpose": "implementation_parity_not_new_independent_validation",
               "compared_at_utc": _utc_now(), "device": str(engine.device), "precision": "float32_without_amp",
               "historical_device": "CUDA Tesla T4 + AMP; lote 64",
               "n_expected": int(len(table)), "n_checked": int(len(valid)), "new_inference_passes": 1,
               "repeated_images_for_determinism": int(len(repetitions)),
               "score_absolute_tolerance": float(tolerance), "required_class_agreement": 1.0,
               "required_preprocessing_equality": "exact",
               "score_max_abs_diff": float(diff.max()) if len(diff) else None,
               "score_median_abs_diff": float(diff.median()) if len(diff) else None,
               "score_p95_abs_diff": float(diff.quantile(0.95)) if len(diff) else None,
               "score_outside_tolerance": int((~frame.score_within_tolerance).sum()),
               "class_mismatches": int((~frame.class_equal).sum()),
               "preprocessing_mismatches": int((~frame.preprocessing_equal).sum()),
               "errors": int(frame.error.ne("").sum()),
               "repeat_max_abs_diff": float(repetitions.max()) if len(repetitions) else None,
               "elapsed_seconds": time.perf_counter()-start,
               "median_inference_ms": float(valid.inference_ms.median()) if len(valid) else None,
               "environment": {"python": platform.python_version(), "platform": platform.platform(),
                               "torch": torch.__version__, "torchvision": torchvision.__version__,
                               "pillow": PIL.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                               "cpu_threads": torch.get_num_threads()},
               "model_sha256": EXPECTED_MODEL_SHA256, "threshold_changed": False,
               "weights_changed": False, "historical_metrics_changed": False}
    summary["parity_passed"] = bool(len(table) == 624 and len(valid) == len(table)
                                     and frame.preprocessing_equal.all() and frame.class_equal.all()
                                     and frame.score_within_tolerance.all()
                                     and len(repetitions) == 8 and repetitions.max() <= 1e-7)
    _write_json(output_dir / "resumen_paridad.json", summary)
    if len(valid) == len(table):
        # Vectores de regresión para probar la instalación local, no una nueva
        # muestra de evaluación clínica. Incluye los casos cercanos al umbral.
        selected_ids = table.sort_values(["distance_to_threshold", "file_id"]).file_id.head(4).tolist()
        for case in ("FN", "FP", "TN", "TP"):
            for fid in table.loc[table.case_type.eq(case)].sort_values("file_id").file_id.head(3):
                if fid not in selected_ids:
                    selected_ids.append(fid)
        for fid in table.sort_values("file_id").file_id:
            if len(selected_ids) >= 16:
                break
            if fid not in selected_ids:
                selected_ids.append(fid)
        portable_dir = output_dir.parent / "pruebas_portables"
        (portable_dir / "imagenes").mkdir(parents=True, exist_ok=True)
        fixtures = []
        current = valid.set_index("file_id")
        by_id = table.set_index("file_id")
        for fid in selected_ids[:16]:
            item = by_id.loc[fid]
            source = image_path(data_root, item.relative_path)
            relative = f"imagenes/{fid}{source.suffix.lower()}"
            destination = portable_dir / relative
            shutil.copyfile(source, destination)
            tensor = preprocess_image(load_image(source))
            tensor_sha = hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()
            fixtures.append({"file_id": fid, "path": relative, "original_relative_path": item.relative_path,
                             "image_sha256": item.byte_sha256, "tensor_sha256": tensor_sha,
                             "score_cpu_kaggle": float(current.loc[fid, "score_fase4"]),
                             "predicted_cpu_kaggle": int(current.loc[fid, "predicted_fase4"])})
        _write_json(portable_dir / "casos.json", {
            "purpose": "technical_regression_only_not_clinical_evaluation", "n": len(fixtures),
            "selection": "4 nearest historical threshold + 3 per confusion category; fill by file_id",
            "source": "Chest X-Ray Images (Pneumonia), test oficial ya evaluado",
            "source_url": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
            "phase4_kaggle_parity_passed": summary["parity_passed"],
            "score_absolute_tolerance": 0.0001, "tensor_equality": "sha256_exact",
            "model_sha256": EXPECTED_MODEL_SHA256, "threshold": float(engine.threshold),
            "environment_cpu_kaggle": summary["environment"], "cases": fixtures})
    return summary


def write_phase4_summary(project_root, parity_summary=None):
    root = Path(project_root)
    verify_preservation(root)
    if parity_summary is None:
        status, passed = "PENDIENTE_KAGGLE", None
    else:
        passed = bool(parity_summary["parity_passed"])
        status = "PARIDAD_APROBADA" if passed else "PARIDAD_INCOMPLETA"
    catalog = pd.read_csv(root / "evidencia/errores_test.csv", keep_default_na=False)
    n_reviewed = int(catalog.estado_revision_humana.str.strip().str.casefold().eq("revisado").sum())
    human_status = "completa" if n_reviewed == len(catalog) else "parcial" if n_reviewed else "pendiente"
    state = {"status": status, "parity_passed": passed, "updated_at_utc": _utc_now(),
             "test_error_catalog": {"false_negatives": 23, "false_positives": 33,
                                    "visual_human_review": human_status, "n_reviewed": n_reviewed},
             "model_preserved": True, "calibrated": False, "patient_identity_verified": False,
             "new_training_performed": False, "new_independent_validation": False,
             "parity": parity_summary}
    _write_json(root / "evidencia/estado_fase4.json", state)
    panels = sorted((root / "evidencia/paneles_errores").glob("*.jpg"))
    panel_html = "".join(f'<figure><img src="paneles_errores/{html.escape(x.name)}" '
                         f'alt="{html.escape(x.stem)}"><figcaption>{html.escape(x.name)}</figcaption></figure>'
                         for x in panels)
    note = ("La comparación con radiografías originales aún debe ejecutarse en Kaggle."
            if passed is None else "Los criterios técnicos de paridad se cumplieron en esta ejecución."
            if passed else "La paridad no está completa. Revisa los casos de discrepancia antes de afirmar equivalencia.")
    parity_text = ("No ejecutada con radiografías originales en este paquete inicial."
                   if parity_summary is None else html.escape(json.dumps(parity_summary, ensure_ascii=False, indent=2)))
    document = f'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Fase 4 · Evidencia de integración</title><style>body{{font:16px/1.6 system-ui,sans-serif;color:#173145;background:#f4f7fa;margin:0}}
main{{max-width:1100px;margin:auto;padding:36px}}h1,h2{{line-height:1.2}}.card,figure{{background:white;padding:24px;border-radius:14px;margin:20px 0}}
img{{width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}a{{color:#007c83}}.label{{color:#007c83;font-weight:700}}
@media print{{body{{background:white}}main{{padding:0}}figure{{break-inside:avoid}}}}</style><main>
<p class="label">GRUPO 2 · (Juan)⁴ · PROTOTIPO ACADÉMICO</p><h1>Fase 4: inferencia y revisión de errores</h1>
<section class="card"><h2>{html.escape(status)}</h2><p>{note}</p>
<p>El modelo, las etiquetas, el preprocesamiento y el umbral 0,874506533 se conservan. Esta fase no entrena ni calibra.</p></section>
<section class="card"><h2>Resultado histórico de fase 3</h2><p>624 imágenes; 568 aciertos; 23 falsos negativos y 33 falsos positivos.
Sensibilidad 94,10 %, especificidad 85,90 %, exactitud 91,03 % y AUROC 0,9653. Son resultados del mismo test, no validación externa.</p>
<p><a href="errores_test.csv">Catálogo de los 56 errores</a> · <a href="preservacion_experimento.json">Conservación de archivos originales</a></p></section>
<section class="card"><h2>Cómo registrar la revisión</h2><p>Abre errores_test.csv. Completa estado_revision_humana, observacion_visual,
limitacion_o_hipotesis, revisor y fecha_revision. Describe lo observable; una causa clínica requiere valoración experta. No cambies etiquetas,
particiones, umbral o modelo a partir de esta revisión del test. El catálogo inicial mantiene la revisión humana como pendiente.</p></section>
<section class="card"><h2>Comprobación de implementación</h2><p>Tensor idéntico al preprocesamiento original; diferencia absoluta de score ≤0,01;
100 % de clases coincidentes. La tolerancia se fijó antes de comparar, por el cambio de GPU con AMP a CPU FP32; no es una tolerancia clínica.</p>
<pre>{parity_text}</pre></section><h2>Paneles de errores</h2><p>Imagen original orientada y entrada real de 224 × 224. Las etiquetas y scores son históricos.</p>
{panel_html or '<p>Los paneles se generarán en Kaggle cuando estén disponibles las radiografías originales.</p>'}
<section class="card"><h2>Límites del prototipo</h2><p>Los puntajes no están calibrados. No se ha verificado independencia clínica de pacientes ni validación externa.
Validar un JPEG/PNG no determina que sea una radiografía de tórax. La aplicación no guarda automáticamente las imágenes cargadas.</p></section></main></html>'''
    (root / "evidencia/informe_fase4.html").write_text(document, encoding="utf-8")
    return state


def export_bundle(project_root, zip_path):
    root, zip_path = Path(project_root).resolve(), Path(zip_path).resolve()
    if zip_path.is_relative_to(root):
        raise ValueError("Guarda el ZIP fuera de la carpeta del proyecto.")
    verify_preservation(root)
    def selected(path):
        relative = path.relative_to(root)
        return (path.is_file() and not any(x in {"__pycache__", ".venv", ".git", ".ipynb_checkpoints"}
                                          for x in relative.parts)
                and path.suffix not in {".pyc", ".pyo"} and relative.as_posix() != "indice_fase4_sha256.json")
    files = [x for x in sorted(root.rglob("*")) if selected(x)]
    _write_json(root / "indice_fase4_sha256.json", {x.relative_to(root).as_posix(): sha256_file(x) for x in files})
    files.append(root / "indice_fase4_sha256.json")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, (Path("neumonia_app") / path.relative_to(root)).as_posix())
    return zip_path
