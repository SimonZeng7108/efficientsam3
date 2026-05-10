from pathlib import Path

from PIL import Image

from fewshot_lora.images import ImageResolutionError, resolve_image_path


def _touch_image(path: Path, size=(8, 6)) -> None:
    Image.new("RGB", size, color=(255, 0, 0)).save(path)


def test_resolve_image_path_prefers_exact_filename(tmp_path: Path):
    image = tmp_path / "part.jpg.bmp"
    _touch_image(image)

    resolved = resolve_image_path(tmp_path, "part.jpg.bmp")

    assert resolved.path == image
    assert resolved.width == 8
    assert resolved.height == 6
    assert resolved.match_kind == "exact"


def test_resolve_image_path_matches_case_insensitively(tmp_path: Path):
    image = tmp_path / "Part.JPG.BMP"
    _touch_image(image)

    resolved = resolve_image_path(tmp_path, "part.jpg.bmp")

    assert resolved.path == image
    assert resolved.match_kind == "case_insensitive"


def test_resolve_image_path_matches_unique_compound_stem(tmp_path: Path):
    image = tmp_path / "Part.JPG.BMP"
    _touch_image(image)

    resolved = resolve_image_path(tmp_path, "Part")

    assert resolved.path == image
    assert resolved.match_kind == "stem"


def test_resolve_image_path_reports_ambiguous_stem_matches(tmp_path: Path):
    _touch_image(tmp_path / "Part.JPG.BMP")
    _touch_image(tmp_path / "Part.PNG")

    try:
        resolve_image_path(tmp_path, "Part")
    except ImageResolutionError as exc:
        assert exc.kind == "ambiguous"
        assert "Part.JPG.BMP" in exc.message
        assert "Part.PNG" in exc.message
    else:
        raise AssertionError("expected ambiguous image resolution error")
