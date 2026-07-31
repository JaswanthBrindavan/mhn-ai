"""Translating our JSON Schema into Gemini's dialect.

Our schemas are written provider-neutrally: nullable fields are a `["string", "null"]`
union and objects close with `additionalProperties: false`, both of which Anthropic accepts.
Gemini takes an OpenAPI-3 subset that rejects a type union and does not know
`additionalProperties`. Getting this wrong cost a live run: the request was refused, the
error was wrapped as transient, and every document retried.
"""

from app.integrations.ai.gemini_provider import to_gemini_schema
from app.services.extraction import EXTRACTION_JSON_SCHEMA


def test_nullable_union_becomes_type_plus_nullable():
    assert to_gemini_schema({"type": ["string", "null"]}) == {"type": "string", "nullable": True}


def test_plain_type_is_untouched():
    assert to_gemini_schema({"type": "string"}) == {"type": "string"}


def test_additional_properties_is_dropped():
    out = to_gemini_schema({"type": "object", "additionalProperties": False, "properties": {}})
    assert "additionalProperties" not in out
    assert out["type"] == "object"


def test_translation_reaches_nested_properties_and_array_items():
    out = to_gemini_schema(
        {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"unit": {"type": ["string", "null"]}},
                    },
                }
            },
        }
    )
    item = out["properties"]["rows"]["items"]
    assert item["properties"]["unit"] == {"type": "string", "nullable": True}
    assert "additionalProperties" not in item


def test_required_lists_of_strings_survive():
    out = to_gemini_schema({"type": "object", "required": ["a", "b"], "properties": {}})
    assert out["required"] == ["a", "b"]


def test_the_real_extraction_schema_translates_cleanly():
    """The whole point: our actual schema must come out Gemini-legal."""
    out = to_gemini_schema(EXTRACTION_JSON_SCHEMA)

    def walk(node):
        if isinstance(node, dict):
            assert not isinstance(node.get("type"), list), f"union type survived: {node}"
            assert "additionalProperties" not in node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(out)
    result_props = out["properties"]["results"]["items"]["properties"]
    assert result_props["test_name"] == {"type": "string"}
    assert result_props["reference_range"] == {"type": "string", "nullable": True}
