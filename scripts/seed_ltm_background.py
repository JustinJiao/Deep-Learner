from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.ltm import LTM


def _build_entries() -> list[dict]:
    return [
        {
            "key": "bg_scope_docs",
            "type": "background",
            "score": 1.0,
            "content": (
                "Project document scope includes exactly three filings: "
                "Amazon 10K 2024.pdf, Alphabet 10K 2024.pdf, and MSFT 10-K.pdf."
            ),
        },
        {
            "key": "bg_company_aliases",
            "type": "background",
            "score": 1.0,
            "content": (
                "Company and alias mapping: Amazon.com, Inc. (Amazon, AMZN); "
                "Alphabet Inc. (Alphabet, Google, GOOG, GOOGL); "
                "Microsoft Corporation (Microsoft, MSFT). "
                "In this corpus, references to Google company map to Alphabet filing entity."
            ),
        },
        {
            "key": "bg_coreference_company_group",
            "type": "background",
            "score": 0.98,
            "content": (
                "Coreference normalization for group mentions: "
                "'these companies', 'all three companies', and 'the three companies' "
                "should expand to Amazon, Alphabet, and Microsoft."
            ),
        },
        {
            "key": "bg_fiscal_year_anchors",
            "type": "background",
            "score": 1.0,
            "content": (
                "Latest reported fiscal year across the three filings is 2024. "
                "Fiscal year end anchors are company-specific: "
                "Amazon fiscal year ended December 31, 2024; "
                "Alphabet fiscal year ended December 31, 2024; "
                "Microsoft fiscal year ended June 30, 2024."
            ),
        },
        {
            "key": "bg_cloud_terms_amazon_alphabet",
            "type": "background",
            "score": 0.97,
            "content": (
                "Cloud term anchors for Amazon and Alphabet: "
                "Amazon Web Services (AWS) is Amazon's cloud business and appears as a reportable segment. "
                "Alphabet cloud business appears as Google Cloud segment. "
                "Google Cloud revenue is described as coming from Google Cloud Platform and Google Workspace offerings."
            ),
        },
        {
            "key": "bg_cloud_terms_microsoft",
            "type": "background",
            "score": 0.97,
            "content": (
                "Cloud term anchors for Microsoft: "
                "Azure is Microsoft's cloud services platform. "
                "Intelligent Cloud is a Microsoft reportable segment including Azure and other cloud services. "
                "Microsoft Cloud revenue metric includes Azure and other cloud services, Office 365 Commercial, "
                "the commercial portion of LinkedIn, Dynamics 365, and other commercial cloud properties."
            ),
        },
        {
            "key": "bg_cross_company_scope_note",
            "type": "background",
            "score": 0.92,
            "content": (
                "Comparison caveat: Microsoft Cloud is a broader metric than AWS and Google Cloud segments, "
                "so cross-company cloud revenue comparisons should note scope differences."
            ),
        },
        {
            "key": "bg_table_reading_anchor",
            "type": "background",
            "score": 0.9,
            "content": (
                "Financial table reading anchor: in filing tables, amounts are often labeled as 'in millions' "
                "and must be interpreted with the correct year header and company context."
            ),
        },
    ]


def main() -> None:
    ltm = LTM()
    entries = _build_entries()

    inserted = ltm.upsert(entries)
    print(f"seeded_background_entries={inserted}")
    print("seeded_keys=", [entry["key"] for entry in entries])


if __name__ == "__main__":
    main()
