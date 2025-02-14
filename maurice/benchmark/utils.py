import re
from typing import Iterable


def try_convert(x: str) -> str | int | float:
    """Try to convert a string to an integer or a float"""

    x = x.strip()

    if x.isnumeric():
        return int(x)
    elif x.replace(".", "", 1).isnumeric():
        return float(x)

    return x


def decompose_name(model_str: str) -> Iterable[tuple[str, list[str | int | float]]]:
    """Turns strings of the form `fn1(arg1, arg2,..)->fn2(arg1, arg2, ...)->...`
    into an interator of the form `[("fn1", [arg1, arg2, ...]), ("fn2, [arg1,
    arg2, ...]), ...]`"""

    matches = re.findall(r"(\w+)\(([^()]*)\)", model_str)

    for match in matches:
        function_name = match[0]
        arguments = list(map(try_convert, match[1].split(",")))
        yield function_name, arguments


def to_hf_name(model_name: str) -> str:
    """Converts a model name `fn1(arg1, arg2,..)->fn2(arg1, arg2, ...)->...` to
    a model string that can be used as a huggingface model id"""

    return (
        model_name.replace(">", "")
        .replace(".", "_")
        .replace("(", ".")
        .replace(",", ".")
        .replace(")", "")
        .strip("-.")
    )
