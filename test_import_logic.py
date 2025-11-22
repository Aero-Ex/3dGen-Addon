
import os
from pathlib import Path
import time

# Mock setup
output_dir = Path.home() / "Documents" / "TRELLIS_Output"
output_dir.mkdir(parents=True, exist_ok=True)

# Create dummy files with different timestamps
(output_dir / "old.glb").touch()
time.sleep(0.1)
(output_dir / "newer.obj").touch()
time.sleep(0.1)
(output_dir / "newest_upscaled.ply").touch()

# Test logic
extensions = ['*.glb', '*.obj', '*.ply', '*.stl']
generated_files = []

for ext in extensions:
    generated_files.extend(list(output_dir.glob(ext)))

if not generated_files:
    print("No files found")
else:
    latest_file = max(generated_files, key=lambda p: p.stat().st_mtime)
    print(f"Latest file: {latest_file.name}")

# Cleanup
(output_dir / "old.glb").unlink()
(output_dir / "newer.obj").unlink()
(output_dir / "newest_upscaled.ply").unlink()
