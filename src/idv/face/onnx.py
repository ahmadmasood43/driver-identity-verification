from pathlib import Path


def session(path: Path, device: str = "cuda"):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        # Loads the CUDA/cuDNN libraries that the torch cu126 wheels install.
        ort.preload_dlls(directory="")
        providers.insert(0, "CUDAExecutionProvider")
    sess = ort.InferenceSession(str(path), providers=providers)
    if device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA provider unavailable for {path}")
    return sess
