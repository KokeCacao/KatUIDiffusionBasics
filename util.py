from typing import Any, Dict, Optional

from backend import variable


def should_update(
    target_fps: int,
    start_timestamp: float,
    start_render_variable: Optional[Dict[str, Any]],
    sid: str,
    current_timestamp: float,
    current_render_variable: Optional[Dict[str, Any]],
) -> bool:
    if current_render_variable is None:
        current_render_variable = variable.ip2render[variable.sid2ip[sid]]
        if current_render_variable is None:
            return False # no render variable
    render_variables_changed = start_render_variable is None or \
        (start_render_variable["camera"] != current_render_variable["camera"] \
            or start_render_variable["fov"] != current_render_variable["fov"] \
                or start_render_variable["near"] != current_render_variable["near"] \
                    or start_render_variable["far"] != current_render_variable["far"])
    time_change = (current_timestamp - start_timestamp) > 1.0 / target_fps
    return render_variables_changed or time_change
