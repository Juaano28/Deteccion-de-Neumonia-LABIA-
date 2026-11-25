"""Predicción de una imagen local usando el mismo motor de Streamlit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from neumonia.inference import InferenceEngine, InferenceError, load_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Clasificación académica NORMAL/PNEUMONIA con MobileNetV2.")
    parser.add_argument("image", type=Path, help="Ruta de una imagen JPEG o PNG.")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "model")
    parser.add_argument("--device", default="cpu", help="cpu (predeterminado), cuda o auto.")
    parser.add_argument("--output", type=Path, help="Guardar solamente el resultado JSON en esta ruta.")
    args = parser.parse_args()
    try:
        image = load_image(args.image)
        engine = InferenceEngine(args.model_dir, device=args.device)
        result = engine.predict_image(image)
        serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
    except (InferenceError, OSError, RuntimeError) as exc:
        print(f"No se pudo completar la predicción: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
