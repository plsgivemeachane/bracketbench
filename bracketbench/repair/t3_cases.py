"""Curated T3 case set: broken Jupyter Notebooks (CONTEXT.md -> T3).

Each case is the text of a .ipynb with real notebook-level problems: corrupt cell ``source``,
mangled ``outputs``, cell-type inconsistency, and (in one case) text that is not JSON at
all. ``VALID_NOTEBOOK`` is the fully-valid baseline the broken cases are broken relative to;
it doubles as the reference repair for tests and for whole-text-replacement stub scripts.
The broken cases are built from explicit dicts and serialized with ``json.dumps`` so the
JSON escaping is always correct.

The static-check scoring of each default case (equal default weights) is exercised in
``tests/test_t3.py``.
"""

import json
from typing import List

_VALID_NOTEBOOK_DICT = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {},
    "cells": [
        {
            "cell_type": "code",
            "id": "cell-1",
            "metadata": {},
            "source": "print('hello')",
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": ["hello\n"]}
            ],
            "execution_count": 1,
        },
        {
            "cell_type": "markdown",
            "id": "cell-2",
            "metadata": {},
            "source": "# Title",
        },
    ],
}

VALID_NOTEBOOK = json.dumps(_VALID_NOTEBOOK_DICT)

# Corrupt cell `source`: an int instead of a string/list of strings.
CORRUPT_SOURCE_NOTEBOOK = json.dumps(
    {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "id": "cell-1",
                "metadata": {},
                "source": 42,
                "outputs": [],
                "execution_count": 1,
            }
        ],
    }
)

# Cell-type inconsistency: a markdown cell carries an `outputs` key it must not have.
OUTPUTS_ON_MARKDOWN_NOTEBOOK = json.dumps(
    {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "markdown",
                "id": "cell-1",
                "metadata": {},
                "source": "# Title",
                "outputs": [],
            }
        ],
    }
)

# Mangled output: an execute_result output missing its required `data`.
MANGLED_OUTPUT_NOTEBOOK = json.dumps(
    {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "id": "cell-1",
                "metadata": {},
                "source": "1 + 1",
                "outputs": [
                    {"output_type": "execute_result", "metadata": {}, "execution_count": 1}
                ],
                "execution_count": 1,
            }
        ],
    }
)

# Missing `execution_count` on a code cell.
MISSING_EXECUTION_COUNT_NOTEBOOK = json.dumps(
    {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "id": "cell-1",
                "metadata": {},
                "source": "print('hello')",
                "outputs": [],
            }
        ],
    }
)

# Invalid `cell_type`: "CODE" is not in {code, markdown, raw}.
INVALID_CELL_TYPE_NOTEBOOK = json.dumps(
    {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "CODE",
                "id": "cell-1",
                "metadata": {},
                "source": "print('hello')",
            }
        ],
    }
)

# Not JSON at all: an unterminated notebook document (the T3 prerequisite-gate case).
UNPARSEABLE_NOTEBOOK = '{"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": ['

DEFAULT_T3_CASES: List[str] = [
    CORRUPT_SOURCE_NOTEBOOK,
    OUTPUTS_ON_MARKDOWN_NOTEBOOK,
    MANGLED_OUTPUT_NOTEBOOK,
    MISSING_EXECUTION_COUNT_NOTEBOOK,
    INVALID_CELL_TYPE_NOTEBOOK,
    UNPARSEABLE_NOTEBOOK,
]
