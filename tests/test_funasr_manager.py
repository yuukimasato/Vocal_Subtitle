from vocal_subtitle.asr.funasr_manager import find_local_model


def test_find_local_model_detects_modelscope_double_dash_layout(tmp_path):
    model = (
        tmp_path
        / "models"
        / "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    )
    model.mkdir(parents=True)
    (model / "model.pt").write_bytes(b"weights")

    assert find_local_model(
        "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        tmp_path,
    ) == model


def test_find_local_model_returns_modelscope_snapshot(tmp_path):
    model = (
        tmp_path
        / "models"
        / "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    )
    snapshot = model / "snapshots" / "master"
    snapshot.mkdir(parents=True)
    (snapshot / "configuration.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.pt").write_bytes(b"weights")

    assert find_local_model(
        "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        tmp_path,
    ) == snapshot
