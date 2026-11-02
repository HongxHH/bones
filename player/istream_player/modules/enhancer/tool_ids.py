# -*- coding: utf-8 -*-
"""Multi-type enhance tool IDs (DESIGN: workspace/multi_enhance/DESIGN.md)."""

TOOL_NONE = 0
TOOL_SR_LOW = 1
TOOL_SR_MED = 2
TOOL_SR_HIGH = 3
TOOL_SR_X2_MED = 4
TOOL_RESTORE_SHARP = 5

N_TOOLS = 6

TOOL_NAMES = {
    TOOL_NONE: "none",
    TOOL_SR_LOW: "sr_low",
    TOOL_SR_MED: "sr_med",
    TOOL_SR_HIGH: "sr_high",
    TOOL_SR_X2_MED: "sr_x2_med",
    TOOL_RESTORE_SHARP: "restore_sharp",
}

TOOL_KINDS = {
    TOOL_NONE: "none",
    TOOL_SR_LOW: "sr",
    TOOL_SR_MED: "sr",
    TOOL_SR_HIGH: "sr",
    TOOL_SR_X2_MED: "sr",
    TOOL_RESTORE_SHARP: "restore",
}

# download_action index → legal tools (BBB ladder 240/360/480/720/1080).
LEGAL_BY_DOWNLOAD = {
    0: {TOOL_NONE, TOOL_SR_LOW, TOOL_SR_MED, TOOL_SR_HIGH},
    1: {TOOL_NONE, TOOL_SR_LOW, TOOL_SR_MED, TOOL_SR_HIGH},
    2: {TOOL_NONE, TOOL_SR_LOW, TOOL_SR_MED, TOOL_SR_HIGH},
    3: {TOOL_NONE, TOOL_SR_X2_MED, TOOL_RESTORE_SHARP},
    4: {TOOL_NONE, TOOL_RESTORE_SHARP},
}


def tool_name(tool_id: int) -> str:
    return TOOL_NAMES.get(int(tool_id), f"tool_{tool_id}")


def is_native_res_tool(tool_id: int) -> bool:
    """Tools that may run when repr is not in classic SR resolution_set (720/1080)."""
    return int(tool_id) in (TOOL_SR_X2_MED, TOOL_RESTORE_SHARP)
