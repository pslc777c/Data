import os
import hashlib
import json

root = os.environ.get("CI_ARTIFACTS_DIR", "ci_artifacts")

manifest = []


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

for base, _, files in os.walk(root):
    for f in files:
        p = os.path.join(base, f)
        try:
            st = os.stat(p)
            h = md5_file(p)
            manifest.append({
                "path": os.path.relpath(p, root),
                "bytes": st.st_size,
                "md5": h
            })
        except Exception as e:
            manifest.append({
                "path": os.path.relpath(p, root),
                "error": str(e)
            })

manifest.sort(key=lambda x: x.get("path", ""))

out = os.path.join(root, "snapshot_manifest.json")

with open(out, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, ensure_ascii=False)

print(f"Wrote {out} with {len(manifest)} items")
