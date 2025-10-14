#!/usr/bin/env python3
"""
Quick test of the prefill data system.
Tests analyzer and generator without running full benchmark.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from benchmarks.loader import BenchmarkLoader
from generation.prefill_analyzer import PrefillDataAnalyzer
from generation.prefill_generator import PrefillDataGenerator


def test_prefill_system():
    """Test prefill analyzer and generator."""

    print("🧪 Testing Prefill Data System\n")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set in environment")
        print("   Please set it in .env file or export it")
        return False

    # Load benchmark
    print("\n1. Loading benchmark...")
    loader = BenchmarkLoader()
    try:
        benchmark = loader.load("benchmarks/v1.json")
        print(f"   ✓ Loaded: {benchmark.name}")
        case = benchmark.cases[0]
        print(f"   ✓ Behavior: {case.behavior[:80]}...")
    except Exception as e:
        print(f"   ❌ Failed to load benchmark: {e}")
        return False

    # Test analyzer
    print("\n2. Testing Prefill Analyzer...")
    try:
        analyzer = PrefillDataAnalyzer(model="gpt-5")
        spec = analyzer.analyze(case.behavior, case.pages)
        print(f"   ✓ Generated spec with {len(spec.entities)} entity type(s)")
        for entity in spec.entities:
            print(f"      - {entity.type} on {entity.page}: {entity.count} item(s)")
    except Exception as e:
        print(f"   ❌ Analyzer failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test generator
    print("\n3. Testing Prefill Generator...")
    try:
        generator = PrefillDataGenerator(model="gpt-5")
        prefill_data = generator.generate(spec, case.behavior)
        print(f"   ✓ Generated data for {len(prefill_data.data_by_page)} page(s)")
        for page, items in prefill_data.data_by_page.items():
            print(f"      - {page}: {len(items)} item(s)")
            if items and len(items) > 0:
                first_item = items[0]
                keys = list(first_item.keys())[:3]
                print(f"        Sample keys: {', '.join(keys)}")
    except Exception as e:
        print(f"   ❌ Generator failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Save results
    print("\n4. Saving results...")
    try:
        test_dir = Path("test_output")
        test_dir.mkdir(exist_ok=True)

        spec_path = test_dir / "test_prefill_spec.json"
        import json

        spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")
        print(f"   ✓ Spec saved: {spec_path}")

        data_path = test_dir / "test_prefill_data.json"
        prefill_data.save(data_path)
        print(f"   ✓ Data saved: {data_path}")
    except Exception as e:
        print(f"   ⚠️  Failed to save results: {e}")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nYou can inspect the generated files in test_output/")
    return True


if __name__ == "__main__":
    success = test_prefill_system()
    sys.exit(0 if success else 1)
