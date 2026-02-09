import base64, os, pathlib

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
models = ROOT / "models"
b64_path = models / "hybrid_model.pkl.b64"
out_path = models / "hybrid_model.pkl"

if not b64_path.exists():
    raise SystemExit(f"Missing: {b64_path}")

models.mkdir(parents=True, exist_ok=True)
data = base64.b64decode(b64_path.read_text(encoding="utf-8"))
out_path.write_bytes(data)
print(f"✅ Restored model: {out_path} ({out_path.stat().st_size/1024/1024:.2f} MB)")
