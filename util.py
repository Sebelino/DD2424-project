import json
from typing import Any


def json_dumps(dct: dict):
    return json.dumps(dct, indent=4, sort_keys=True)


def dumps_inline_lists(obj: Any, indent: int = 2) -> str:
    def _format(o: Any, level: int) -> str:
        sp = ' ' * (indent * level)
        if isinstance(o, dict):
            items = []
            for i, (k, v) in enumerate(o.items()):
                key = json.dumps(k)
                val = _format(v, level + 1)
                comma = ',' if i < len(o) - 1 else ''
                # if the value is multiline, leave it as-is
                if '\n' in val:
                    items.append(f"{sp}{' ' * indent}{key}: {val}{comma}")
                else:
                    items.append(f"{sp}{' ' * indent}{key}: {val}{comma}")
            return "{\n" + "\n".join(items) + f"\n{sp}" + "}"
        elif isinstance(o, list):
            # render list inline
            elems = []
            for item in o:
                if isinstance(item, (dict, list)):
                    elems.append(_format(item, level + 1))
                else:
                    elems.append(json.dumps(item))
            return "[ " + ", ".join(elems) + " ]"
        else:
            return json.dumps(o)

    return _format(obj, 0)


def suppress_weights_only_warning():
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"You are using `torch\.load` with `weights_only=False`.*",
        category=FutureWarning
    )


def shorten_label(label, limit=50):
    if len(label) > limit:
        label = label[:limit - 1] + "…"
    return label
