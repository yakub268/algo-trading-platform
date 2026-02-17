"""
Quick structure test for Computer Vision Bot (no heavy dependencies)

Tests the basic structure and integration without requiring OpenCV/tesseract.
"""

import os
import sys
import inspect

# Add project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_orchestrator_integration():
    """Test that the CV bot is properly integrated in the orchestrator"""
    print("Testing orchestrator integration...")

    try:
        from master_orchestrator import BOT_REGISTRY

        # Check if CV bot is in registry
        cv_bot_found = False
        for bot_config in BOT_REGISTRY:
            if "Computer-Vision-Bot" in bot_config.name:
                cv_bot_found = True
                print(f"✅ CV Bot found in registry: {bot_config.name}")
                print(f"  Module: {bot_config.module_path}")
                print(f"  Class: {bot_config.class_name}")
                print(f"  Market: {bot_config.market.value}")
                print(f"  Enabled: {bot_config.enabled}")
                print(
                    f"  Schedule: {bot_config.schedule_type} ({bot_config.schedule_value})"
                )
                print(f"  Allocation: {bot_config.allocation_pct*100:.1f}%")
                break

        if not cv_bot_found:
            print("❌ CV Bot not found in orchestrator registry")
            return False

        return True

    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist"""
    print("\nTesting file structure...")

    required_files = [
        "bots/computer_vision_bot.py",
        "config/computer_vision_config.py",
        "scripts/setup_cv_bot.py",
        "tests/test_computer_vision_bot.py",
        "templates/README.md",
        "COMPUTER_VISION_BOT.md",
    ]

    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_exist = False

    return all_exist


def test_import_structure():
    """Test that the module structure is correct (without heavy dependencies)"""
    print("\nTesting import structure...")

    try:
        # Read the file and check class structure
        with open("bots/computer_vision_bot.py", "r") as f:
            content = f.read()

        required_classes = [
            "ComputerVisionBot",
            "WindowsMCPClient",
            "ComputerVisionEngine",
            "BrokerInterfaceManager",
        ]

        for class_name in required_classes:
            if f"class {class_name}" in content:
                print(f"✅ {class_name} class found")
            else:
                print(f"❌ {class_name} class not found")
                return False

        # Check for required methods
        required_methods = ["run_strategy", "get_status", "find_opportunities"]

        for method_name in required_methods:
            if f"def {method_name}" in content:
                print(f"✅ {method_name} method found")
            else:
                print(f"❌ {method_name} method not found")
                return False

        return True

    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        return False


def test_config_structure():
    """Test configuration structure"""
    print("\nTesting configuration structure...")

    try:
        # Import config without dependencies
        config_path = "config/computer_vision_config.py"
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False

        with open(config_path, "r") as f:
            content = f.read()

        required_configs = [
            "BROKER_CONFIGS",
            "TEMPLATE_MAPPINGS",
            "COLOR_RANGES",
            "CVConfig",
        ]

        for config_name in required_configs:
            if config_name in content:
                print(f"✅ {config_name} found in config")
            else:
                print(f"❌ {config_name} not found in config")
                return False

        return True

    except Exception as e:
        print(f"❌ Config structure test failed: {e}")
        return False


def main():
    """Run all structure tests"""
    print("=" * 60)
    print("COMPUTER VISION BOT STRUCTURE TEST")
    print("=" * 60)

    tests = [
        ("File Structure", test_file_structure),
        ("Import Structure", test_import_structure),
        ("Config Structure", test_config_structure),
        ("Orchestrator Integration", test_orchestrator_integration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All structure tests passed! CV Bot is properly integrated.")
        print("\nNext steps:")
        print("1. Run: python scripts\\setup_cv_bot.py")
        print("2. Install dependencies with the setup script")
        print("3. Create template images for broker interfaces")
        print("4. Test with: python bots\\computer_vision_bot.py --test")
    else:
        print("❌ Some structure tests failed. Please check the issues above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
