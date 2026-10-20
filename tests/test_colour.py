"""Colour difference: the conversion to CIELAB and CIEDE2000 against published values, and
the comparison of two pictures of the same framing."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest
from PIL import Image, ImageCms

from conftest import CLI
from modellab import colour

P3 = "/System/Library/ColorSync/Profiles/Display P3.icc"
MLX_ENV = os.environ.get("MODELLAB_TEST_MLX_ENV")

# Sharma, Wu and Dalal (2005), "The CIEDE2000 Color-Difference Formula: Implementation Notes,
# Supplementary Test Data, and Mathematical Observations", Table 1: Lab pair and expected value.
SHARMA = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0011, -2.4900), 4.7461),
    ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize("first, second, expected", SHARMA)
def test_ciede2000_matches_the_published_test_data(first, second, expected):
    assert colour.ciede2000(numpy.array(first), numpy.array(second)) == pytest.approx(expected, abs=1e-4)


def test_srgb_converts_to_cielab_with_signed_a_and_b():
    lab = colour.srgb_to_lab(numpy.array([[[255, 0, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]], numpy.uint8))
    assert lab[0, 0] == pytest.approx([53.2408, 80.0925, 67.2032], abs=0.01)
    assert lab[0, 1] == pytest.approx([32.2970, 79.1875, -107.8602], abs=0.01)  # a negative b, not wrapped to ~255
    assert lab[0, 2] == pytest.approx([100, 0, 0], abs=0.01)
    assert lab[0, 3] == pytest.approx([0, 0, 0], abs=0.01)


def test_pictures_are_compared_as_seen_and_only_when_framed_alike(tmp_path):
    grey = Image.new("RGB", (300, 200), (120, 120, 120))
    grey.save(tmp_path / "a.png")
    grey.resize((150, 100)).save(tmp_path / "small.png")  # same framing, another size
    Image.new("RGB", (300, 200), (130, 120, 110)).save(tmp_path / "warm.png")
    assert colour.compare(tmp_path / "a.png", tmp_path / "small.png")["mean"] == pytest.approx(0, abs=1e-6)
    warm = colour.compare(tmp_path / "a.png", tmp_path / "warm.png")
    assert 3 < warm["mean"] < 12 and warm["p95"] == pytest.approx(warm["mean"]) and warm["cie76_mean"] > warm["mean"]
    Image.new("RGB", (200, 200), (120, 120, 120)).save(tmp_path / "square.png")
    with pytest.raises(ValueError, match="framed differently"):
        colour.compare(tmp_path / "a.png", tmp_path / "square.png")
    # A picture tagged with another profile (Display P3 here) is converted to sRGB, never read as if it were sRGB.
    p3 = ImageCms.getOpenProfile(str(P3))
    tagged = Image.new("RGB", (300, 200), (200, 110, 60))
    tagged.save(tmp_path / "p3.png", icc_profile=p3.tobytes())
    ImageCms.profileToProfile(tagged, p3, ImageCms.createProfile("sRGB"), outputMode="RGB").save(tmp_path / "converted.png")
    assert colour.compare(tmp_path / "p3.png", tmp_path / "converted.png")["mean"] < 0.5
    tagged.save(tmp_path / "relabelled.png")  # the same numbers read as sRGB: what a relabel would do
    assert colour.compare(tmp_path / "relabelled.png", tmp_path / "converted.png")["mean"] > 2

@pytest.mark.skipif(not MLX_ENV, reason="set MODELLAB_TEST_MLX_ENV to an installed mlx environment")
def test_the_colour_command_reports_each_pair(lab, tmp_path):
    (lab.root / "envs").mkdir(parents=True)
    (lab.root / "envs" / "mlx").symlink_to(Path(MLX_ENV))
    Image.new("RGB", (300, 200), (120, 120, 120)).save(tmp_path / "a.png")
    Image.new("RGB", (300, 200), (130, 120, 110)).save(tmp_path / "b.png")
    result = subprocess.run([sys.executable, CLI, "colour", str(tmp_path / "a.png"), str(tmp_path / "b.png"), "--json"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["method"] == "CIEDE2000" and 3 < report["mean"] < 12
