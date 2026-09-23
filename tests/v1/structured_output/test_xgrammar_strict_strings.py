# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the strict length-bounded JSON strings in XgrammarBackend.

xgrammar compiles a JSON-schema string with minLength/maxLength into a
repetition of ``[^\"\\\r\n]``: raw control characters (TAB, 0x01, ...) pass the
mask and make the output invalid JSON, and escapes (``\\"``, ``\\t``) are
impossible. XgrammarBackend._compile_json_schema rewrites that character rule.

Tests drive the real XgrammarBackend.compile_grammar with a real tokenizer
(``XSS_TOKENIZER``, default Qwen/Qwen3-0.6B) and check the resulting
GrammarMatcher. ``XSS_BACKEND=/path/to/unpatched/backend_xgrammar.py`` runs the
same tests against another copy of the backend (red).
"""

import importlib.util
import json
import os
import types

import pytest
import transformers  # noqa: F401  (import order: avoids a circular import)

import vllm.v1.structured_output.backend_xgrammar as _live
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

MODEL = os.environ.get("XSS_TOKENIZER", "Qwen/Qwen3-0.6B")


def _load_backend_module():
    path = os.environ.get("XSS_BACKEND")
    if not path:
        return _live
    spec = importlib.util.spec_from_file_location("xss_backend_xgrammar", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bx = _load_backend_module()

NAME = {"type": "string", "minLength": 2, "maxLength": 12}
SCHEMA = {
    "type": "object",
    "properties": {
        "name": NAME,
        "notes": {"type": "array", "items": {"type": "string", "maxLength": 8}},
        "plain": {"type": "string"},
        "sku": {"type": "string", "pattern": "^[A-Z]{3}-[0-9]{4}$"},
        "n": {"type": "integer"},
    },
    "required": ["name", "notes", "plain", "sku", "n"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def backend():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    cfg = types.SimpleNamespace(
        structured_outputs_config=types.SimpleNamespace(disable_any_whitespace=False),
        speculative_config=types.SimpleNamespace(num_speculative_tokens=3),
    )
    return bx.XgrammarBackend(cfg, tokenizer=tok, vocab_size=len(tok))


def _accepts(backend, schema, text: str) -> bool:
    g = backend.compile_grammar(StructuredOutputOptions.JSON, json.dumps(schema))
    ok = g.matcher.accept_string(text)
    return bool(ok) and g.matcher.is_completed()


def _doc(name='"Comte"', notes='["nutty"]', plain='"x"', sku='"ABC-1234"', n="1"):
    return f'{{"name": {name}, "notes": {notes}, "plain": {plain}, "sku": {sku}, "n": {n}}}'


def test_valid_document_accepted(backend):
    assert _accepts(backend, SCHEMA, _doc())


@pytest.mark.parametrize("ch", ["\t", "\x01", "\x1f", "\x0b"])
def test_raw_control_char_in_bounded_string_rejected(backend, ch):
    assert not _accepts(backend, SCHEMA, _doc(name=f'"Co{ch}mte"'))


def test_raw_tab_in_bounded_array_item_rejected(backend):
    assert not _accepts(backend, SCHEMA, _doc(notes='["nu\tty"]'))


@pytest.mark.parametrize("esc", ['\\"', "\\t", "\\\\", "\\u00e9", "\\n"])
def test_escapes_in_bounded_string_rejected_like_stock(backend, esc):
    # Amended 2026-09-23 (on-call): the escape branch inside the bounded
    # repetition cost 0.5-2.7 s per mask fill and stalled the engine; bounded
    # strings keep stock xgrammar's no-escape behaviour, minus control chars.
    assert not _accepts(backend, SCHEMA, _doc(name=f'"Co{esc}mte"'))


def test_length_bounds_still_enforced(backend):
    assert not _accepts(backend, SCHEMA, _doc(name='"C"'))  # minLength 2
    assert _accepts(backend, SCHEMA, _doc(name='"' + "C" * 12 + '"'))
    assert not _accepts(backend, SCHEMA, _doc(name='"' + "C" * 13 + '"'))


def test_unicode_counts_as_one_char(backend):
    assert _accepts(backend, SCHEMA, _doc(name='"' + "é" * 12 + '"'))


def test_unbounded_and_pattern_strings_unchanged(backend):
    assert not _accepts(backend, SCHEMA, _doc(plain='"a\tb"'))
    assert _accepts(backend, SCHEMA, _doc(plain='"a\\tb"'))
    assert not _accepts(backend, SCHEMA, _doc(sku='"AB-1234"'))


def test_schema_without_bounded_strings_uses_stock_compile(backend):
    s = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    assert _accepts(backend, s, '{"a": "x"}')
    if hasattr(bx, "fix_bounded_json_string_chars"):
        import xgrammar as xgr

        ebnf = str(xgr.Grammar.from_json_schema(json.dumps(s)))
        assert bx.fix_bounded_json_string_chars(ebnf) is None


def test_rewrite_only_touches_bounded_char_rules():
    import xgrammar as xgr

    assert hasattr(bx, "fix_bounded_json_string_chars"), "patch not applied"
    ebnf = str(xgr.Grammar.from_json_schema(json.dumps(SCHEMA)))
    fixed = bx.fix_bounded_json_string_chars(ebnf)
    a, b = ebnf.splitlines(), fixed.splitlines()
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(diff) == 2  # name chars, notes item chars
    assert all("syv_json_string_escape" not in y for _, y in diff)
    assert len(b) == len(a)


def test_bounded_string_mask_cost_stays_fast(backend):
    """Regression (2026-09-23 engine stalls): filling the mask inside a long
    bounded string must stay stock-fast. With the escape alternation it grew to
    0.5-2.7 s per token at 40-130 chars into a maxLength-200 string."""
    import time

    import xgrammar as xgr

    s = {"type": "object", "properties": {"c": {"type": "string", "maxLength": 200}},
         "required": ["c"], "additionalProperties": False}
    g = backend.compile_grammar(StructuredOutputOptions.JSON, json.dumps(s))
    tok = backend.tokenizer
    ids = tok.encode('{"c": "' + "A deeply nutty, brothy wheel with a long caramel finish; " * 3,
                     add_special_tokens=False)
    bm = xgr.allocate_token_bitmask(1, backend.vocab_size)
    worst = 0.0
    for t in ids:
        t0 = time.perf_counter()
        g.matcher.fill_next_token_bitmask(bm, 0)
        worst = max(worst, time.perf_counter() - t0)
        assert g.matcher.accept_token(t)
    assert worst < 0.05, f"worst mask fill {worst * 1e3:.0f} ms"


def test_loadgen_schemas_compile_and_accept_valid_instances(backend):
    """The cheese-lab schemas (single cheese, shop with $defs/$ref/anyOf/pattern,
    scorecard) still compile and accept a hand-written valid instance."""
    cheese = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 2, "maxLength": 80},
            "country": {"type": "string", "minLength": 2, "maxLength": 60},
            "milk": {"type": "string", "enum": ["cow", "goat"]},
            "aging_days": {"type": "integer", "minimum": 0, "maximum": 3650},
            "tasting_notes": {
                "type": "array",
                "items": {"type": "string", "maxLength": 40},
                "minItems": 1,
                "maxItems": 5,
            },
        },
        "required": ["name", "country", "milk", "aging_days", "tasting_notes"],
        "additionalProperties": False,
    }
    doc = (
        '{\n  "name": "Rogue River Blue",\n  "country": "USA",\n  "milk": "cow",'
        '\n  "aging_days": 300,\n  "tasting_notes": ["fruity", "boozy pear"]\n}'
    )
    assert _accepts(backend, cheese, doc)
    assert not _accepts(backend, cheese, doc.replace("boozy pear", "boozy\tpear"))


def test_fallback_on_rewrite_failure(backend, monkeypatch):
    if not hasattr(bx, "fix_bounded_json_string_chars"):
        pytest.skip("patch not applied")
    monkeypatch.setattr(bx, "fix_bounded_json_string_chars", lambda e: "root ::= (")
    # Broken rewrite -> stock compile (raw TAB accepted again, but no crash).
    assert _accepts(backend, SCHEMA, _doc())
