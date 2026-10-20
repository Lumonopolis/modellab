"""How different two pictures of the same framing look: CIEDE2000 in CIELAB (D65), per pixel.

Pictures are read as people see them. One tagged with another colour profile (Display P3,
for example) is converted to sRGB first, never relabelled. The two must show the same framing,
such as two renders of one crop: a per-pixel difference between different crops means
nothing. A different aspect is refused here; a different crop of the same aspect cannot be
told from the pixels, so making sure of the framing is the caller's part.

Lab values are computed here from the sRGB formulas, with signed a and b. Pillow's own LAB
mode stores a and b unsigned, so a small negative value reads as about 255.

Runs inside the ModelLab environment (numpy, Pillow): `modellab colour` starts it.
"""

import io
import json
import sys
from pathlib import Path

import numpy
from PIL import Image, ImageCms, ImageOps

WHITE = numpy.array([0.95047, 1.0, 1.08883])  # D65
TO_XYZ = numpy.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
ASPECT = 0.01  # framings closer than this count as the same


def srgb_to_lab(rgb):
    """CIELAB (L 0..100, signed a and b) for an sRGB array of 0..255 values, shape (..., 3)."""
    linear = numpy.asarray(rgb, dtype=numpy.float64) / 255
    linear = numpy.where(linear <= 0.04045, linear / 12.92, ((linear + 0.055) / 1.055) ** 2.4)
    xyz = linear @ TO_XYZ.T / WHITE
    f = numpy.where(xyz > (6 / 29) ** 3, numpy.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return numpy.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def ciede2000(lab1, lab2):
    """CIEDE2000 difference between Lab arrays of shape (..., 3), with kL = kC = kH = 1."""
    L1, a1, b1 = numpy.moveaxis(numpy.asarray(lab1, dtype=numpy.float64), -1, 0)
    L2, a2, b2 = numpy.moveaxis(numpy.asarray(lab2, dtype=numpy.float64), -1, 0)
    C = (numpy.hypot(a1, b1) + numpy.hypot(a2, b2)) / 2
    G = 0.5 * (1 - numpy.sqrt(C ** 7 / (C ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = numpy.hypot(a1p, b1), numpy.hypot(a2p, b2)
    h1p = numpy.degrees(numpy.arctan2(b1, a1p)) % 360
    h2p = numpy.degrees(numpy.arctan2(b2, a2p)) % 360
    neutral = C1p * C2p == 0
    dh = h2p - h1p
    dh = numpy.where(dh > 180, dh - 360, numpy.where(dh < -180, dh + 360, dh))
    dh = numpy.where(neutral, 0, dh)
    dL, dC = L2 - L1, C2p - C1p
    dH = 2 * numpy.sqrt(C1p * C2p) * numpy.sin(numpy.radians(dh / 2))
    Lbar, Cbar = (L1 + L2) / 2, (C1p + C2p) / 2
    hsum = h1p + h2p
    hbar = numpy.where(numpy.abs(h1p - h2p) > 180, numpy.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2), hsum / 2)
    hbar = numpy.where(neutral, hsum, hbar)
    T = (1 - 0.17 * numpy.cos(numpy.radians(hbar - 30)) + 0.24 * numpy.cos(numpy.radians(2 * hbar))
         + 0.32 * numpy.cos(numpy.radians(3 * hbar + 6)) - 0.20 * numpy.cos(numpy.radians(4 * hbar - 63)))
    SL = 1 + 0.015 * (Lbar - 50) ** 2 / numpy.sqrt(20 + (Lbar - 50) ** 2)
    SC = 1 + 0.045 * Cbar
    SH = 1 + 0.015 * Cbar * T
    RT = (-2 * numpy.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7))
          * numpy.sin(numpy.radians(60 * numpy.exp(-(((hbar - 275) / 25) ** 2)))))
    return numpy.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH))


def load_srgb(path):
    """The picture as sRGB, upright (EXIF), converted from its own profile when it has one."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        profile = image.info.get("icc_profile")
        image = image.convert("RGB")
        if profile:
            source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
            image = ImageCms.profileToProfile(image, source, ImageCms.createProfile("sRGB"), outputMode="RGB")
        return image


def compare(first, second):
    """CIEDE2000 mean and 95th percentile (and CIE76 mean) between two pictures of one framing.
    The second is resized to the first; framings that differ are refused."""
    a, b = load_srgb(first), load_srgb(second)
    if abs(a.width / a.height - b.width / b.height) > ASPECT * a.width / a.height:
        raise ValueError(f"{first} ({a.width}x{a.height}) and {second} ({b.width}x{b.height}) are framed differently; "
                         "a per-pixel colour difference needs the same framing")
    lab_a, lab_b = srgb_to_lab(numpy.asarray(a)), srgb_to_lab(numpy.asarray(b.resize(a.size, Image.Resampling.LANCZOS)))
    each = ciede2000(lab_a, lab_b)
    return {"method": "CIEDE2000", "mean": float(each.mean()), "p95": float(numpy.percentile(each, 95)),
            "cie76_mean": float(numpy.linalg.norm(lab_a - lab_b, axis=-1).mean()), "size": list(a.size)}


def main(request):
    """{"pairs": [[first, second], ...]} -> one report per pair."""
    reports = []
    for first, second in request["pairs"]:
        try:
            reports.append({"first": first, "second": second, **compare(Path(first), Path(second))})
        except ValueError as error:
            reports.append({"first": first, "second": second, "error": str(error)})
    return reports


if __name__ == "__main__":
    print(json.dumps(main(json.loads(sys.argv[1]))))
