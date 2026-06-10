import site
from pathlib import Path
import os
from subprocess import check_call

build_dir = Path("/build/geco2")

if not build_dir.exists():
    cwd = os.getcwd()
    print("GeCo2: Additional installation required. This may take up to 5 minutes.", flush=True)

    check_call(["cp", "-r", "/cache/geco2/models/ops", build_dir])
    os.chdir(build_dir)
    
    check_call(["/cache/geco2/.venv/bin/python", "setup.py", "build"])
    
    target = build_dir / "dist"
    target.mkdir()
    check_call(["uv", "--project", "/cache/geco2", "pip", "install", "--no-build-isolation", "--target", target, "."])

    site.addsitedir(target)

    from models.counter_infer import CNT
    CNT(zero_shot = True, image_size = 1024, num_objects = 3, emb_dim = 256, kernel_dim = 3, reduction = 16)

    os.chdir(cwd)
    print("Activation complete.", flush=True)
