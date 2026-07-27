import json

import pytest

from scripts.evaluate_acoustic_gold import (
    evaluate_boundaries,
    load_word_boundaries,
    main,
    schema_path,
)


def _payload(words):
    return {"schema_version": "acoustic-gold-v1", "words": words}


def _word(word_id, onset, offset, speaker="A"):
    return {
        "id": word_id,
        "text": word_id,
        "onset": onset,
        "offset": offset,
        "speaker": speaker,
    }


def test_load_rejects_duplicate_word_ids(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(_payload([_word("w1", 0.1, 0.2), _word("w1", 0.3, 0.4)])),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_word_boundaries(path)


def test_repository_schema_is_present():
    assert schema_path().is_file()
    schema = json.loads(schema_path().read_text(encoding="utf-8"))
    assert schema["$defs"]["word"]["required"] == [
        "id", "text", "speaker", "onset", "offset"
    ]


def test_evaluate_reports_boundary_and_speaker_metrics():
    gold = {
        "w1": _word("w1", 1.0, 1.5),
        "w2": _word("w2", 2.0, 2.5, speaker="B"),
    }
    prediction = {
        "w1": _word("w1", 1.1, 1.4),
        "w2": _word("w2", 2.0, 2.5, speaker="A"),
    }

    report = evaluate_boundaries(gold, prediction)

    assert report["matched_word_count"] == 2
    assert report["boundary_median_ms"] == pytest.approx(50.0)
    assert report["cut_head_over_threshold_rate"] == 0.5
    assert report["cut_tail_over_threshold_rate"] == 0.5
    assert report["speaker_error_rate"] == 0.5
    assert report["gold_word_coverage"] == 1.0


def test_ci_returns_nonzero_when_gold_word_is_missing(tmp_path):
    gold = tmp_path / "gold.json"
    prediction = tmp_path / "prediction.json"
    gold.write_text(
        json.dumps(_payload([_word("w1", 0.1, 0.2), _word("w2", 0.3, 0.4)])),
        encoding="utf-8",
    )
    prediction.write_text(
        json.dumps(_payload([_word("w1", 0.1, 0.2)])), encoding="utf-8"
    )

    assert main(["--gold", str(gold), "--prediction", str(prediction), "--ci"]) == 3
