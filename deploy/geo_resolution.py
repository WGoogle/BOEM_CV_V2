"""
Extract spatial resolution from GeoTIFF metadata

Fail safe is a hardcoded value for meters_per_pixel of .05 
"""
from __future__ import annotations
import logging
import math
import struct
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)

def extract_meters_per_pixel(filepath, fallback):
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

    if 34264 in tags:
        mpp = _mpp_from_transformation_tag(tags, filepath.name)
        if mpp is not None:
            return mpp
    if 33550 in tags:
        mpp = _mpp_from_pixel_scale_tag(tags, filepath.name)
        if mpp is not None:
            return mpp

    logger.info(
        f"{filepath.name}: no geo resolution tags found — using fallback ({fallback})"
    )
    return fallback

def extract_geo_metadata(filepath, fallback_mpp):
    # Return a dict with meters_per_pixel, lat/lon origin, and CRS type.
    # Values are None when the corresponding tag is absent.
    filepath = Path(filepath)
    result = {
        "meters_per_pixel": fallback_mpp,
        "mpp_source": "fallback",
        "latitude": None,
        "longitude": None,
        "crs_type": None,
    }
    if filepath.suffix.lower() not in (".tif", ".tiff"):
        return result

    try:
        tags = _read_tiff_tags(filepath)
    except Exception as exc:
        logger.warning(f"{filepath.name}: failed to read TIFF tags — {exc}")
        return result

    result["crs_type"] = "geographic" if _is_geographic_crs(tags) else "projected"

    lat = _get_latitude(tags)
    lon = _get_longitude(tags)
    if lat is not None:
        result["latitude"] = lat
    if lon is not None:
        result["longitude"] = lon

    if 34264 in tags:
        mpp = _mpp_from_transformation_tag(tags, filepath.name)
        if mpp is not None:
            result["meters_per_pixel"] = mpp
            result["mpp_source"] = "ModelTransformationTag"
            return result
    if 33550 in tags:
        mpp = _mpp_from_pixel_scale_tag(tags, filepath.name)
        if mpp is not None:
            result["meters_per_pixel"] = mpp
            result["mpp_source"] = "ModelPixelScaleTag"
            return result
    return result

def compute_corner_coords(geo, height_px, width_px):
    """Return labeled lat/lon for the 4 image corners, or None values if unknown.

    Assumes the mosaic is north-up: the origin lat/lon is the top-left corner,
    and pixel rows increase southward. Works for both geographic (deg) and
    projected (metre) CRSs; geographic uses the same 111,320 m/deg approximation
    used elsewhere in this file.
    """
    empty = {
        "top_left":     {"latitude": None, "longitude": None},
        "top_right":    {"latitude": None, "longitude": None},
        "bottom_left":  {"latitude": None, "longitude": None},
        "bottom_right": {"latitude": None, "longitude": None},
    }

    lat0 = geo.get("latitude")
    lon0 = geo.get("longitude")
    mpp  = geo.get("meters_per_pixel")
    if lat0 is None or lon0 is None or mpp is None or mpp <= 0:
        return empty

    width_m  = width_px  * mpp
    height_m = height_px * mpp

    if geo.get("crs_type") == "geographic":
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
        if m_per_deg_lon <= 0:
            return empty
        dlat = height_m / m_per_deg_lat
        dlon = width_m  / m_per_deg_lon
        top_left     = (lat0,        lon0)
        top_right    = (lat0,        lon0 + dlon)
        bottom_left  = (lat0 - dlat, lon0)
        bottom_right = (lat0 - dlat, lon0 + dlon)
    else:
        # Projected CRS — the stored "latitude"/"longitude" are already metres
        # in that projection; step by width/height in the same units.
        top_left     = (lat0,            lon0)
        top_right    = (lat0,            lon0 + width_m)
        bottom_left  = (lat0 - height_m, lon0)
        bottom_right = (lat0 - height_m, lon0 + width_m)

    return {
        "top_left":     {"latitude": top_left[0],     "longitude": top_left[1]},
        "top_right":    {"latitude": top_right[0],    "longitude": top_right[1]},
        "bottom_left":  {"latitude": bottom_left[0],  "longitude": bottom_left[1]},
        "bottom_right": {"latitude": bottom_right[0], "longitude": bottom_right[1]},
    }


def _get_longitude(tags):
    # From ModelTransformationTag: tx = matrix[3]
    if 34264 in tags:
        doubles = _unpack_doubles(tags[34264], 16)
        lon = doubles[3]
        if -180 <= lon <= 180:
            return lon
    # From ModelTiepointTag (33922): doubles[3] is X (lon)
    if 33922 in tags:
        bo, dtype, count, data = tags[33922]
        if count >= 6:
            doubles = _unpack_doubles(tags[33922], 6)
            lon = doubles[3]
            if -180 <= lon <= 180:
                return lon
    return None

# TIFF type sizes (bytes per element)
_TYPE_SIZE = {
    1: 1, 2: 1, 3: 2, 4: 4, 5: 8,    # BYTE, ASCII, SHORT, LONG, RATIONAL
    6: 1, 7: 1, 8: 2, 9: 4, 10: 8,   # SBYTE, UNDEFINED, SSHORT, SLONG, SRATIONAL
    11: 4, 12: 8,                       # FLOAT, DOUBLE
}

def _read_tiff_tags(filepath):
    # Read IFD0 tags from a TIFF file
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

def _unpack_doubles(tag_entry, n):
    bo, dtype, count, data = tag_entry
    return [struct.unpack(bo + "d", data[i * 8 : (i + 1) * 8])[0] for i in range(n)]

def _unpack_shorts(tag_entry, n):
    bo, dtype, count, data = tag_entry
    return [struct.unpack(bo + "H", data[i * 2 : (i + 1) * 2])[0] for i in range(n)]

def _get_latitude(tags):
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

def _is_geographic_crs(tags):
    if 34735 not in tags:
        return True  # assume geographic if we can't tell
    shorts = _unpack_shorts(tags[34735], tags[34735][2])
    # GeoKey 1024 = GTModelTypeGeoKey; value 2 = Geographic
    for i in range(4, len(shorts), 4):
        key_id = shorts[i] if i < len(shorts) else 0
        if key_id == 1024:
            value = shorts[i + 3] if i + 3 < len(shorts) else 0
            return value == 2 
    return True 

def _mpp_from_transformation_tag(tags, name):
    # Compute m/px from the 4×4 ModelTransformationTag
    doubles = _unpack_doubles(tags[34264], 16)
    # 4×4 row-major: [a, b, 0, tx,  e, f, 0, ty,  ...]
    a, b = doubles[0], doubles[1]
    e, f = doubles[4], doubles[5]

    if _is_geographic_crs(tags):
        deg_per_px_x = math.sqrt(a ** 2 + b ** 2)
        deg_per_px_y = math.sqrt(e ** 2 + f ** 2)

        lat = _get_latitude(tags)
        if lat is None:
            lat = 0.0  

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

def _mpp_from_pixel_scale_tag(tags, name):
    # Compute m/px from ModelPixelScaleTag (ScaleX, ScaleY, ScaleZ)
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