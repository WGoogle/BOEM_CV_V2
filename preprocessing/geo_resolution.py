"""
Extract spatial resolution from GeoTIFF metadata

Reads the ModelTransformationTag (tag 34264) or ModelPixelScaleTag
(tag 33550) from a TIFF file and computes meters_per_pixel using the
embedded coordinate reference system.

Fail safe is a hardcoded value for meters_per_pixel. 

Reason why this is an important addition is because on the researcher handoff i saw that it was 
.05 meters/pixel which was warned that this varies, so I thought since we have all the geodata
in the .tif files, might as well use them to improve our preprocessing. 

"""
from __future__ import annotations

import logging
import math
import struct

from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def extract_meters_per_pixel(filepath: Path, fallback: float | None = None) -> Optional[float]:
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix not in (".tif", ".tiff"):
        logger.debug(f"{filepath.name}: not a TIFF — using fallback ({fallback})")
        return fallback

    try:
        tags = _read_tiff_tags(filepath)
    except Exception as exc:
        logger.warning(f"{filepath.name}: failed to read TIFF tags — {exc}")
        return fallback

    #  ModelTransformationTag (34264)
    if 34264 in tags:
        mpp = _mpp_from_transformation_tag(tags, filepath.name)
        if mpp is not None:
            return mpp

    # ModelPixelScaleTag (33550)
    if 33550 in tags:
        mpp = _mpp_from_pixel_scale_tag(tags, filepath.name)
        if mpp is not None:
            return mpp

    logger.info(
        f"{filepath.name}: no geo resolution tags found — using fallback ({fallback})"
    )
    return fallback

# TIFF type sizes (bytes per element)
_TYPE_SIZE = {
    1: 1, 2: 1, 3: 2, 4: 4, 5: 8,    # BYTE, ASCII, SHORT, LONG, RATIONAL
    6: 1, 7: 1, 8: 2, 9: 4, 10: 8,   # SBYTE, UNDEFINED, SSHORT, SLONG, SRATIONAL
    11: 4, 12: 8,                       # FLOAT, DOUBLE
}


def _read_tiff_tags(filepath: Path) -> dict:
    """Read IFD0 tags from a TIFF file.  Returns {tag_id: raw_bytes}."""
    with open(filepath, "rb") as f:
        header = f.read(4)
        bo = "<" if header[:2] == b"II" else ">"
        magic = struct.unpack(bo + "H", header[2:4])[0]
        if magic != 42:
            raise ValueError(f"Not a TIFF file (magic={magic})")

        ifd_offset = struct.unpack(bo + "I", f.read(4))[0]
        f.seek(ifd_offset)
        n_entries = struct.unpack(bo + "H", f.read(2))[0]

        tags: dict[int, tuple[str, int, bytes]] = {}
        for _ in range(n_entries):
            tag_id = struct.unpack(bo + "H", f.read(2))[0]
            dtype = struct.unpack(bo + "H", f.read(2))[0]
            count = struct.unpack(bo + "I", f.read(4))[0]
            value_raw = f.read(4)

            elem_size = _TYPE_SIZE.get(dtype, 1)
            total_bytes = count * elem_size

            if total_bytes <= 4:
                data = value_raw[:total_bytes]
            else:
                offset = struct.unpack(bo + "I", value_raw)[0]
                pos = f.tell()
                f.seek(offset)
                data = f.read(total_bytes)
                f.seek(pos)

            tags[tag_id] = (bo, dtype, count, data)

    return tags


def _unpack_doubles(tag_entry: tuple, n: int) -> list[float]:
    """Unpack *n* IEEE-754 doubles from a tag entry."""
    bo, dtype, count, data = tag_entry
    return [struct.unpack(bo + "d", data[i * 8 : (i + 1) * 8])[0] for i in range(n)]


def _unpack_shorts(tag_entry: tuple, n: int) -> list[int]:
    bo, dtype, count, data = tag_entry
    return [struct.unpack(bo + "H", data[i * 2 : (i + 1) * 2])[0] for i in range(n)]


def _get_latitude(tags: dict) -> Optional[float]:
    """Try to extract a reference latitude from the geo metadata."""
    # From ModelTransformationTag: ty = matrix[7] is the latitude origin
    if 34264 in tags:
        doubles = _unpack_doubles(tags[34264], 16)
        lat = doubles[7]
        if -90 <= lat <= 90:
            return lat

    # From ModelTiepointTag (33922): 6 values [I, J, K, X, Y, Z], Y is lat
    if 33922 in tags:
        bo, dtype, count, data = tags[33922]
        if count >= 6:
            doubles = _unpack_doubles(tags[33922], 6)
            lat = doubles[4]
            if -90 <= lat <= 90:
                return lat

    return None


def _is_geographic_crs(tags: dict) -> bool:
    """Check if GeoKeyDirectoryTag indicates a geographic (degree) CRS."""
    if 34735 not in tags:
        return True  # assume geographic if we can't tell
    shorts = _unpack_shorts(tags[34735], tags[34735][2])
    # GeoKey 1024 = GTModelTypeGeoKey; value 2 = Geographic
    for i in range(4, len(shorts), 4):
        key_id = shorts[i] if i < len(shorts) else 0
        if key_id == 1024:
            value = shorts[i + 3] if i + 3 < len(shorts) else 0
            return value == 2  # ModelTypeGeographic
    return True  # default assumption


def _mpp_from_transformation_tag(tags: dict, name: str) -> Optional[float]:
    """Compute m/px from the 4×4 ModelTransformationTag."""
    doubles = _unpack_doubles(tags[34264], 16)
    # 4×4 row-major: [a, b, 0, tx,  e, f, 0, ty,  ...]
    a, b = doubles[0], doubles[1]
    e, f = doubles[4], doubles[5]

    if _is_geographic_crs(tags):
        # Units are degrees — convert to metres
        deg_per_px_x = math.sqrt(a ** 2 + b ** 2)
        deg_per_px_y = math.sqrt(e ** 2 + f ** 2)

        lat = _get_latitude(tags)
        if lat is None:
            lat = 0.0  # equator fallback

        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))

        mpp_x = deg_per_px_x * m_per_deg_lon
        mpp_y = deg_per_px_y * m_per_deg_lat
        mpp = (mpp_x + mpp_y) / 2.0

        logger.info(
            f"{name}: GeoTIFF resolution = {mpp:.6f} m/px "
            f"({mpp * 1000:.2f} mm/px) at lat {lat:.3f}°"
        )
        return mpp
    else:
        # Projected CRS — units are already metres (typically)
        mpp_x = math.sqrt(a ** 2 + b ** 2)
        mpp_y = math.sqrt(e ** 2 + f ** 2)
        mpp = (mpp_x + mpp_y) / 2.0

        logger.info(
            f"{name}: GeoTIFF resolution (projected CRS) = {mpp:.6f} m/px "
            f"({mpp * 1000:.2f} mm/px)"
        )
        return mpp


def _mpp_from_pixel_scale_tag(tags: dict, name: str) -> Optional[float]:
    """Compute m/px from ModelPixelScaleTag (ScaleX, ScaleY, ScaleZ)."""
    doubles = _unpack_doubles(tags[33550], 3)
    sx, sy = doubles[0], doubles[1]

    if _is_geographic_crs(tags):
        lat = _get_latitude(tags)
        if lat is None:
            lat = 0.0

        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))

        mpp_x = sx * m_per_deg_lon
        mpp_y = sy * m_per_deg_lat
        mpp = (mpp_x + mpp_y) / 2.0
    else:
        mpp = (sx + sy) / 2.0

    logger.info(
        f"{name}: GeoTIFF resolution (PixelScaleTag) = {mpp:.6f} m/px "
        f"({mpp * 1000:.2f} mm/px)"
    )
    return mpp