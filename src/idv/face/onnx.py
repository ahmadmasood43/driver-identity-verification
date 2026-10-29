from pathlib import Path


def session(path: Path, device: str = "cuda"):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        # Loads the CUDA/cuDNN libraries that the torch cu126 wheels install.
        ort.preload_dlls(directory="")
        providers.insert(0, "CUDAExecutionProvider")
    opts = ort.SessionOptions()
    # The insightface exports declare batch 1 on outputs, so batched calls warn on every run.
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), opts, providers=providers)
    if device == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA provider unavailable for {path}")
    return sess
