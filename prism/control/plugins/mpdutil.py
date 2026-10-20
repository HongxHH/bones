from __future__ import annotations

import copy
import re
from xml.etree import ElementTree as ET

from ..schemas import LadderRung


def _iso_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds - hours * 3600 - minutes * 60
    if hours:
        return f"PT{hours}H{minutes}M{secs:.3f}S".replace(".000S", "S")
    if minutes:
        return f"PT{minutes}M{secs:.3f}S".replace(".000S", "S")
    return f"PT{secs:.3f}S".replace(".000S", "S")


def build_static_mpd(
    *,
    title: str,
    duration_s: float,
    segment_duration: float,
    ladder: list[LadderRung],
    media: str = "stream$RepresentationID$/segment_$Number$.m4s",
    initialization: str = "stream$RepresentationID$/segment_.mp4",
    timescale: int = 1000,
) -> str:
    max_w = max((r.width for r in ladder), default=0)
    max_h = max((r.height for r in ladder), default=0)
    seg_ticks = int(segment_duration * timescale)
    mpd = ET.Element(
        "MPD",
        {
            "xmlns": "urn:mpeg:dash:schema:mpd:2011",
            "minBufferTime": "PT1.500S",
            "type": "static",
            "mediaPresentationDuration": _iso_duration(duration_s),
            "maxSegmentDuration": _iso_duration(segment_duration),
            "profiles": "urn:mpeg:dash:profile:isoff-live:2011",
        },
    )
    info = ET.SubElement(mpd, "ProgramInformation")
    title_el = ET.SubElement(info, "Title")
    title_el.text = title
    period = ET.SubElement(mpd, "Period", {"duration": _iso_duration(duration_s)})
    aset = ET.SubElement(
        period,
        "AdaptationSet",
        {
            "segmentAlignment": "true",
            "maxWidth": str(max_w),
            "maxHeight": str(max_h),
            "maxFrameRate": "30",
            "par": "16:9",
            "lang": "und",
            "startWithSAP": "1",
            "contentType": "video",
        },
    )
    ET.SubElement(
        aset,
        "SegmentTemplate",
        {
            "media": media,
            "initialization": initialization,
            "timescale": str(timescale),
            "startNumber": "1",
            "duration": str(seg_ticks),
        },
    )
    for rung in ladder:
        ET.SubElement(
            aset,
            "Representation",
            {
                "id": str(rung.id),
                "mimeType": "video/mp4",
                "codecs": "avc1.4D401F",
                "width": str(rung.width),
                "height": str(rung.height),
                "frameRate": "30",
                "sar": "1:1",
                "bandwidth": str(rung.bitrate_kbps * 1000),
            },
        )
    ET.indent(mpd, space=" ")
    xml = ET.tostring(mpd, encoding="unicode")
    return '<?xml version="1.0"?>\n' + xml + "\n"


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def parse_ladder_from_mpd(mpd_xml: str) -> tuple[list[LadderRung], float]:
    content = re.sub(r'xmlns="[^"]+"', "", mpd_xml, count=1)
    root = ET.fromstring(content)
    duration = 0.0
    dur_attr = root.attrib.get("mediaPresentationDuration", "")
    match = re.match(r"^PT(?:(\d+(?:.\d+)?)H)?(?:(\d+(?:.\d+)?)M)?(?:(\d+(?:.\d+)?)S)?$", dur_attr or "")
    if match:
        h, m, s = (float(x) if x else 0.0 for x in match.group(1, 2, 3))
        duration = 3600 * h + 60 * m + s
    ladder: list[LadderRung] = []
    for el in root.iter():
        if _strip_ns(el.tag) != "Representation":
            continue
        try:
            rid = int(el.attrib["id"])
            width = int(el.attrib.get("width", 0))
            height = int(el.attrib.get("height", 0))
            bw = int(el.attrib.get("bandwidth", 0))
        except (KeyError, ValueError):
            continue
        ladder.append(LadderRung(id=rid, width=width, height=height, bitrate_kbps=max(bw // 1000, 1)))
    return ladder, duration


def filter_representations(mpd_xml: str, allowed_ids: list[int] | None) -> str:
    if allowed_ids is None:
        return mpd_xml
    allowed = {str(i) for i in allowed_ids}
    content = re.sub(r'xmlns="[^"]+"', "", mpd_xml, count=1)
    root = ET.fromstring(content)
    for parent in root.iter():
        doomed = [
            child
            for child in list(parent)
            if _strip_ns(child.tag) == "Representation" and child.attrib.get("id") not in allowed
        ]
        for child in doomed:
            parent.remove(child)
    ET.indent(root, space=" ")
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def to_dynamic_mpd(mpd_xml: str, *, minimum_update_period_s: float = 2.0) -> str:
    content = re.sub(r'xmlns="[^"]+"', "", mpd_xml, count=1)
    root = ET.fromstring(content)
    root.attrib["type"] = "dynamic"
    root.attrib["minimumUpdatePeriod"] = _iso_duration(minimum_update_period_s)
    root.attrib.setdefault("availabilityStartTime", "1970-01-01T00:00:00Z")
    ET.indent(root, space=" ")
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def clone_xml(mpd_xml: str) -> str:
    content = re.sub(r'xmlns="[^"]+"', "", mpd_xml, count=1)
    root = ET.fromstring(content)
    copy.deepcopy(root)
    return mpd_xml
