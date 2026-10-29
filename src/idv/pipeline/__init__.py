from idv.pipeline.types import Pipeline


def load_pipeline(name: str) -> Pipeline:
    if name == "fake":
        from idv.pipeline.fake import FakePipeline

        return FakePipeline()
    raise ValueError(f"unknown pipeline {name!r}")
