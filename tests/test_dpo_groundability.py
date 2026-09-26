"""Tests for the model-visible DPO groundability gate."""

from __future__ import annotations

import unittest
from typing import Any

from src.eval.dpo_groundability import (
    assess_case_groundability,
    triage_candidate_difference_groundability,
    triage_gold_sample_groundability,
)


def tool(
    field: str,
    description: str,
    *,
    default: Any = None,
    required: bool = False,
    nested: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = nested or {
        "type": "string",
        "description": description,
    }
    if default is not None:
        schema["default"] = default
    return {
        "type": "function",
        "function": {
            "name": "f",
            "description": "Test function.",
            "parameters": {
                "type": "object",
                "properties": {field: schema},
                "required": [field] if required else [],
            },
        },
    }


def case(
    *,
    query: str,
    path: str,
    diagnostic: str,
    gold: Any,
    pred: Any,
    schema: dict[str, Any],
    gold_present: bool = True,
    pred_present: bool = True,
    gold_arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arguments = gold_arguments if gold_arguments is not None else {path: gold}
    return {
        "source_sample_id": "sample",
        "user_query": query,
        "relevant_tools": [schema],
        "canonical_gold": {
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "f", "arguments": arguments},
                }
            ],
        },
        "argument_diffs": [
            {
                "tool": "f",
                "gold_call_index": 0,
                "pred_call_index": 0,
                "path": path,
                "type": diagnostic,
                "gold_present": gold_present,
                "pred_present": pred_present,
                "gold": gold,
                "pred": pred,
            }
        ],
    }


class DpoGroundabilityTests(unittest.TestCase):
    def test_query_literal_is_grounded(self) -> None:
        value = case(
            query="Start from recipe 70.",
            path="start",
            diagnostic="wrong_value",
            gold=70,
            pred=140,
            schema=tool("start", "Starting recipe index."),
        )
        self.assertTrue(assess_case_groundability(value)["passed"])

    def test_default_does_not_ground_opaque_identifier(self) -> None:
        value = case(
            query="Get Bitcoin supply.",
            path="uuid",
            diagnostic="wrong_value",
            gold="Qwsogvtv82FCd",
            pred="bitcoin",
            schema=tool(
                "uuid",
                "Coin UUID. Defaults to Qwsogvtv82FCd.",
                default="Qwsogvtv82FCd",
            ),
        )
        result = assess_case_groundability(value)
        self.assertFalse(result["passed"])
        self.assertTrue(
            result["diff_assessments"][0]["ungrounded_leaves"][0][
                "looks_opaque"
            ]
        )

    def test_undocumented_private_code_is_not_grounded(self) -> None:
        value = case(
            query="Find Education apps.",
            path="category",
            diagnostic="wrong_value",
            gold="6017",
            pred="6014",
            schema=tool("category", "App Store category code."),
        )
        self.assertFalse(assess_case_groundability(value)["passed"])

    def test_iso_mapping_requires_declared_standard(self) -> None:
        declared = case(
            query="Use the UK region.",
            path="region",
            diagnostic="wrong_value",
            gold="GB",
            pred="UK",
            schema=tool("region", "ISO 3166 alpha-2 country code."),
        )
        generic = case(
            query="Use the UK region.",
            path="region",
            diagnostic="wrong_value",
            gold="GB",
            pred="UK",
            schema=tool("region", "Geolocation code."),
        )
        self.assertTrue(assess_case_groundability(declared)["passed"])
        self.assertFalse(assess_case_groundability(generic)["passed"])

    def test_page_and_limit_deterministically_ground_offset(self) -> None:
        value = case(
            query="Return the second page with a limit of 25.",
            path="offset",
            diagnostic="wrong_value",
            gold=25,
            pred=50,
            schema=tool("offset", "Number of records to skip as an offset."),
            gold_arguments={"limit": 25, "offset": 25},
        )
        result = assess_case_groundability(value)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["diff_assessments"][0]["evidence"],
            ["deterministic_page_limit_to_offset"],
        )

    def test_invented_time_granularity_is_not_grounded(self) -> None:
        value = case(
            query="Return prices for the past hour.",
            path="interval",
            diagnostic="wrong_value",
            gold="1min",
            pred="1h",
            schema=tool("interval", "Time-series interval."),
        )
        self.assertFalse(assess_case_groundability(value)["passed"])

    def test_schema_documented_percentage_hundredths_is_grounded(self) -> None:
        nested = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tax_rate": {
                        "type": "integer",
                        "description": "Tax percentage expressed in hundredths.",
                    }
                },
            },
        }
        value = case(
            query="Apply a tax rate of 8%.",
            path="items",
            diagnostic="nested_structure_mismatch",
            gold=[{"tax_rate": 800}],
            pred=[{"tax_rate": 8}],
            schema=tool("items", "Line items.", nested=nested),
        )
        self.assertTrue(assess_case_groundability(value)["passed"])

    def test_optional_harmful_extra_can_be_omitted(self) -> None:
        value = case(
            query="Find all houses in this area.",
            path="property_type",
            diagnostic="extra_argument",
            gold=None,
            pred="detached",
            schema=tool("property_type", "Optional property type."),
            gold_present=False,
        )
        self.assertTrue(assess_case_groundability(value)["passed"])

    def test_undocumented_empty_string_override_is_not_grounded(self) -> None:
        value = case(
            query="Find dog breeds under 40 pounds.",
            path="name",
            diagnostic="missing_argument",
            gold="",
            pred=None,
            schema=tool(
                "name",
                "Breed name. Defaults to golden retriever.",
                default="golden retriever",
            ),
            pred_present=False,
        )
        self.assertFalse(assess_case_groundability(value)["passed"])

    def test_gold_prescreen_requires_every_argument_to_be_grounded(self) -> None:
        sample = {
            "sample_id": "sample",
            "messages": [{"role": "user", "content": "Use Tokyo and channel CNN."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "description": "Test function.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "channel_id": {
                                    "type": "string",
                                    "description": "Channel identifier.",
                                },
                            },
                            "required": ["city", "channel_id"],
                        },
                    },
                }
            ],
            "assistant": {
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "arguments": {"city": "Tokyo", "channel_id": "opaque1234"},
                        },
                    }
                ],
            },
        }
        result = triage_gold_sample_groundability(sample)
        self.assertEqual(result["decision"], "AUTO_REJECT")
        self.assertEqual(result["reason"], "opaque_or_external_mapping")
        self.assertEqual(result["failed_argument_count"], 1)

    def test_gold_prescreen_accepts_multi_call_literals(self) -> None:
        schema = tool("city", "City name.", required=True)
        sample = {
            "sample_id": "sample",
            "messages": [
                {"role": "user", "content": "Compare weather in Tokyo and Kyoto."}
            ],
            "tools": [schema],
            "assistant": {
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "f", "arguments": {"city": "Tokyo"}},
                    },
                    {
                        "type": "function",
                        "function": {"name": "f", "arguments": {"city": "Kyoto"}},
                    },
                ],
            },
        }
        result = triage_gold_sample_groundability(sample)
        self.assertEqual(result["decision"], "AUTO_ACCEPT")
        self.assertEqual(result["call_type"], "multi")
        self.assertEqual(result["gold_argument_count"], 2)

    def test_gold_prescreen_excludes_text_target(self) -> None:
        sample = {
            "sample_id": "sample",
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [],
            "assistant": {"content": "Hi", "tool_calls": []},
        }
        result = triage_gold_sample_groundability(sample)
        self.assertEqual(result["decision"], "AUTO_REJECT")
        self.assertEqual(result["reason"], "non_tool_call_target")

    def test_gold_prescreen_abstains_on_semantic_unknown(self) -> None:
        sample = {
            "sample_id": "sample",
            "messages": [{"role": "user", "content": "Find flights to New York."}],
            "tools": [tool("airport", "Airport code.", required=True)],
            "assistant": {
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "f", "arguments": {"airport": "JFK"}},
                    }
                ],
            },
        }
        result = triage_gold_sample_groundability(sample)
        self.assertEqual(result["decision"], "SEMANTIC_UNKNOWN")
        self.assertEqual(result["reason"], "requires_semantic_groundability_judgment")

    def test_difference_triage_passes_clean_literal_error(self) -> None:
        value = case(
            query="Start from recipe 70.",
            path="start",
            diagnostic="wrong_value",
            gold=70,
            pred=140,
            schema=tool("start", "Starting recipe index."),
        )
        result = triage_candidate_difference_groundability(value)
        self.assertEqual(result["decision"], "PASS")

    def test_difference_triage_rejects_known_equivalence(self) -> None:
        value = case(
            query="Use London.",
            path="city",
            diagnostic="wrong_value",
            gold="London",
            pred="london",
            schema=tool("city", "City."),
        )
        value["preference_filter"] = {
            "decision": "excluded",
            "basis": "deterministic_equivalence_guard",
        }
        result = triage_candidate_difference_groundability(value)
        self.assertEqual(result["decision"], "REJECT")

    def test_difference_triage_abstains_on_schema_valid_extra(self) -> None:
        value = case(
            query="Find houses.",
            path="property_type",
            diagnostic="extra_argument",
            gold=None,
            pred="detached",
            schema=tool("property_type", "Optional property type."),
            gold_present=False,
        )
        result = triage_candidate_difference_groundability(value)
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertEqual(
            result["reason"],
            "schema_valid_extra_argument_harmfulness_not_rule_proven",
        )

    def test_difference_triage_abstains_on_string_expansion(self) -> None:
        value = case(
            query="Find weather in Berlin.",
            path="city",
            diagnostic="wrong_value",
            gold="Berlin",
            pred="Berlin, Germany",
            schema=tool("city", "City."),
        )
        result = triage_candidate_difference_groundability(value)
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertEqual(
            result["reason"], "possible_semantically_equivalent_string_expansion"
        )

    def test_difference_triage_abstains_on_literal_code_mapping(self) -> None:
        value = case(
            query="Get the gold price.",
            path="symbol",
            diagnostic="wrong_value",
            gold="gold",
            pred="XAU",
            schema=tool("symbol", "Three-letter commodity code."),
        )
        result = triage_candidate_difference_groundability(value)
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertEqual(
            result["reason"],
            "literal_identifier_or_code_mapping_requires_semantic_review",
        )

    def test_difference_triage_abstains_on_coordinate_representation(self) -> None:
        value = case(
            query="Locate 34.0522 N, 118.2437 W.",
            path="query",
            diagnostic="wrong_value",
            gold="34.0522 N, 118.2437 W",
            pred="34.0522, -118.2437",
            schema=tool("query", "Coordinates."),
        )
        result = triage_candidate_difference_groundability(value)
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertEqual(
            result["reason"], "possible_equivalent_numeric_string_representation"
        )

    def test_difference_triage_accepts_explicit_enum_choice(self) -> None:
        schema = tool(
            "size",
            "Size option.",
            nested={"type": "string", "enum": ["small", "medium", "large"]},
        )
        value = case(
            query="Use the large size.",
            path="size",
            diagnostic="wrong_value",
            gold="large",
            pred="small",
            schema=schema,
        )
        result = triage_candidate_difference_groundability(value)
        self.assertEqual(result["decision"], "PASS")

    def test_difference_triage_abstains_on_incidental_numeric_literal(self) -> None:
        value = case(
            query="Use magnitude 2 for Ursa Major.",
            path="constellation",
            diagnostic="wrong_value",
            gold=2,
            pred=1,
            schema=tool("constellation", "Identifier of the constellation."),
        )
        result = triage_candidate_difference_groundability(value)
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertEqual(
            result["reason"],
            "literal_identifier_or_code_mapping_requires_semantic_review",
        )

    def test_difference_triage_abstains_when_rejected_is_schema_default(self) -> None:
        value = case(
            query="Get information about the product.",
            path="act",
            diagnostic="wrong_value",
            gold="product",
            pred="detail",
            schema=tool("act", "API action.", default="detail"),
        )
        result = triage_candidate_difference_groundability(value)
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertEqual(result["reason"], "rejected_matches_schema_default")

    def test_difference_triage_abstains_on_unquoted_free_form_string(self) -> None:
        value = case(
            query="Please bypass the cache.",
            path="cache_mode",
            diagnostic="wrong_value",
            gold="bypass",
            pred="true",
            schema=tool("cache_mode", "Optional cache bypass parameter."),
        )
        result = triage_candidate_difference_groundability(value)
        self.assertEqual(result["decision"], "UNKNOWN")
        self.assertEqual(
            result["reason"], "free_form_string_preference_not_rule_proven"
        )


if __name__ == "__main__":
    unittest.main()
