import numpy as np
import pytest

from manga_translator import manga_translator as mt
from manga_translator.config import Config
from manga_translator.utils import Context, Quadrilateral


def quad(x1, y1, x2, y2, prob=0.5):
    return Quadrilateral(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int64), '', prob)


def test_merge_detection_results_only_adds_unseen_boxes():
    a = quad(0, 0, 100, 40, 0.9)
    b = quad(0, 100, 80, 130, 0.8)
    dup_of_a = quad(5, 2, 98, 38, 0.95)      # same text found by the loose pass
    new_box = quad(200, 200, 260, 230, 0.4)  # only the loose pass found it
    primary = ([a, b], np.zeros((300, 300), np.uint8), None)
    secondary = ([dup_of_a, new_box], np.ones((300, 300), np.uint8) * 10, None)

    kept, mask_raw, _ = mt.merge_detection_results(primary, secondary)

    assert kept == [a, b, new_box]
    assert (mask_raw == 10).all()  # masks max-combined


def test_merge_detection_results_dedupes_within_the_secondary_pass():
    a = quad(0, 0, 10, 10, 0.9)
    s1 = quad(200, 0, 300, 40, 0.6)
    s2 = quad(205, 2, 295, 38, 0.5)  # overlaps s1, lower prob -> dropped
    kept, _, _ = mt.merge_detection_results(([a], None, None), ([s2, s1], None, None))
    assert kept == [a, s1]

    # Higher-prob secondary wins the slot when it comes second in the list.
    s1b = quad(200, 0, 300, 40, 0.7)
    s2b = quad(205, 2, 295, 38, 0.9)
    kept, _, _ = mt.merge_detection_results(([a], None, None), ([s1b, s2b], None, None))
    assert kept == [a, s2b]


@pytest.mark.asyncio
async def test_run_detection_second_pass_only_with_union_threshold(monkeypatch):
    calls = []

    async def fake_dispatch(detector, img, size, text_threshold, box_threshold, *rest):
        calls.append(box_threshold)
        boxes = [quad(0, 0, 100, 40, 0.9)]
        if box_threshold < 0.7:
            boxes.append(quad(200, 200, 260, 230, 0.4))  # loose pass finds more
        return boxes, np.zeros((50, 50), np.uint8), None

    monkeypatch.setattr(mt, 'dispatch_detection', fake_dispatch)
    engine = mt.MangaTranslator.__new__(mt.MangaTranslator)
    engine.device = 'cpu'
    engine.verbose = False
    engine._model_usage_timestamps = {}

    ctx = Context()
    ctx.img_rgb = np.zeros((50, 50, 3), dtype=np.uint8)

    config = Config()
    result = await engine._run_detection(config, ctx)
    assert calls == [0.7]  # default: single pass, behavior unchanged
    assert len(result[0]) == 1

    config.detector.box_threshold_union = 0.4
    result = await engine._run_detection(config, ctx)
    assert calls == [0.7, 0.7, 0.4]  # strict pass again, plus the loose union pass
    assert len(result[0]) == 2  # union added the box only the loose pass found

    # A union threshold not below box_threshold is pointless and skipped.
    calls.clear()
    config.detector.box_threshold_union = 0.7
    await engine._run_detection(config, ctx)
    assert calls == [0.7]
