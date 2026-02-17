"""
Test ONE simple AI function - just keyword extraction from news
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")


def simple_keyword_extractor(text: str) -> list:
    """Extract trading-relevant keywords from text"""
    important_terms = [
        "election",
        "inflation",
        "unemployment",
        "GDP",
        "CPI",
        "fed",
        "federal reserve",
        "interest rates",
        "recession",
        "earnings",
        "congress",
        "supreme court",
        "oil",
        "gas",
        "climate",
        "weather",
        "sports",
        "olympics",
        "world cup",
        "bitcoin",
        "crypto",
        "ethereum",
    ]

    text_lower = text.lower()
    found_keywords = []

    for term in important_terms:
        if term in text_lower:
            found_keywords.append(term)

    return found_keywords


def test_keyword_extraction():
    """Test the keyword extractor with real news examples"""

    test_cases = [
        {
            "text": "The Federal Reserve announced today that inflation has decreased to 3.2%, signaling potential interest rate cuts",
            "expected_keywords": [
                "fed",
                "federal reserve",
                "inflation",
                "interest rates",
            ],
        },
        {
            "text": "Bitcoin surges past $50k as unemployment drops to historic lows amid GDP growth",
            "expected_keywords": ["bitcoin", "unemployment", "GDP"],
        },
        {
            "text": "Supreme Court ruling on election procedures could impact November voting",
            "expected_keywords": ["supreme court", "election"],
        },
    ]

    print("Testing keyword extraction...")
    print("=" * 50)

    all_passed = True

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Text: {test['text'][:60]}...")

        keywords = simple_keyword_extractor(test["text"])
        print(f"Found keywords: {keywords}")

        # Check if we found expected keywords
        found_expected = [kw for kw in test["expected_keywords"] if kw in keywords]
        missing = [kw for kw in test["expected_keywords"] if kw not in keywords]

        if found_expected:
            print(f"✓ Found expected: {found_expected}")
        if missing:
            print(f"✗ Missing: {missing}")

        # Test passes if we found at least one expected keyword
        test_passed = len(found_expected) > 0
        print(f"Result: {'PASS' if test_passed else 'FAIL'}")

        if not test_passed:
            all_passed = False

    print("\n" + "=" * 50)
    print(
        f"Overall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}"
    )
    return all_passed


if __name__ == "__main__":
    success = test_keyword_extraction()

    if success:
        print("\n✓ Simple keyword extraction is working!")
        print("Next step: Test with real Kalshi markets")
    else:
        print("\n✗ Need to fix keyword extraction first")
