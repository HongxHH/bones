from __future__ import annotations

from typing import Any

from .mpdutil import filter_representations, to_dynamic_mpd


class StaticDeliveryAdapter:
    name = "static"
    labels = ["cpu"]
    steps = ["apply_policy"]

    def resolve_mpd(self, *, mpd_xml: str, policy: dict[str, Any], public_base: str, session_id: str, publication_id: str) -> str:
        return mpd_xml


class SessionFilterDeliveryAdapter:
    """Per-session representation filter + optional dynamic MPD."""

    name = "session_filter"
    labels = ["cpu"]
    steps = ["apply_policy"]

    def resolve_mpd(self, *, mpd_xml: str, policy: dict[str, Any], public_base: str, session_id: str, publication_id: str) -> str:
        allowed = policy.get("allowed_representation_ids")
        xml = filter_representations(mpd_xml, allowed)
        if policy.get("mpd_type") == "dynamic":
            xml = to_dynamic_mpd(xml, minimum_update_period_s=float(policy.get("minimum_update_period_s", 2.0)))
        return xml
