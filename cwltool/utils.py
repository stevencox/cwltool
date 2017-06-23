# no imports from cwltool allowed

from typing import Any, Tuple


def aslist(l):  # type: (Any) -> List[Any]
    if isinstance(l, list):
        return l
    else:
        return [l]


def get_feature(self, feature, kwargs):  # type: (Any, Any, Any) -> Tuple[Any, bool]
    for ov in reversed(kwargs.get("overrides", [])):
        if ov.get("overrideTarget") == self.tool_id:
            for t in ov.get("override", []):
                if t["class"] == feature:
                    return (t, True)
    for t in reversed(self.requirements):
        if t["class"] == feature:
            return (t, True)
    for t in reversed(self.hints):
        if t["class"] == feature:
            return (t, False)
    return (None, None)
