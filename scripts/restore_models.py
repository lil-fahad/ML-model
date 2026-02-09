#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Restore ML models from base64-encoded files.
"""
import base64
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS_DIR = ROOT / "models"


def restore_model(name: str) -> bool:
    """Restore a single model from base64."""
    b64_path = MODELS_DIR / f"{name}.pkl.b64"
    out_path = MODELS_DIR / f"{name}.pkl"
    
    if not b64_path.exists():
        print(f"⚠️  Skipping {name}: {b64_path.name} not found")
        return False
    
    try:
        data = base64.b64decode(b64_path.read_text(encoding="utf-8"))
        out_path.write_bytes(data)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"✅ Restored: {out_path.name} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"❌ Failed to restore {name}: {e}")
        return False


def main():
    """Restore all available models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # List of models to restore
    model_names = ["hybrid_model", "enhanced_model"]
    
    restored = 0
    for name in model_names:
        if restore_model(name):
            restored += 1
    
    print(f"\n✅ Restored {restored} model(s)")


if __name__ == "__main__":
    main()
