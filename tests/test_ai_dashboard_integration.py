"""
Test AI Dashboard Integration

This script tests the AI bot endpoints and validates dashboard functionality.
"""

import os
import sys
import json
import requests
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_dashboard_endpoints():
    """Test all AI dashboard endpoints"""

    base_url = "http://localhost:5000"

    # Test endpoints
    endpoints = [
        "/api/ai/sports-bot/status",
        "/api/ai/arbitrage-bot/status",
        "/api/ai/computer-vision/status",
        "/api/ai/prediction-analyzer/status",
        "/api/ai/news-feed/status",
        "/api/ai/combined-metrics",
    ]

    print("Testing AI Dashboard Integration")
    print("=" * 50)

    results = {}

    for endpoint in endpoints:
        try:
            print(f"\nTesting {endpoint}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)

            if response.status_code == 200:
                data = response.json()
                results[endpoint] = {
                    "status": "success",
                    "response_time": response.elapsed.total_seconds(),
                    "data_keys": (
                        list(data.keys()) if isinstance(data, dict) else "non-dict"
                    ),
                    "success_flag": (
                        data.get("success", False) if isinstance(data, dict) else False
                    ),
                }
                print(
                    f"  ✓ Success (HTTP 200) - Response time: {response.elapsed.total_seconds():.2f}s"
                )
                print(
                    f"  ✓ Data keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
                )

                if isinstance(data, dict) and data.get("success"):
                    print(f"  ✓ API reports success")
                elif isinstance(data, dict) and not data.get("success"):
                    print(
                        f"  ⚠ API reports error: {data.get('error', 'Unknown error')}"
                    )

            else:
                results[endpoint] = {
                    "status": "http_error",
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds(),
                }
                print(f"  ✗ HTTP Error {response.status_code}")

        except requests.exceptions.ConnectionError:
            results[endpoint] = {"status": "connection_error"}
            print(f"  ✗ Connection Error - Is dashboard server running?")

        except requests.exceptions.Timeout:
            results[endpoint] = {"status": "timeout"}
            print(f"  ✗ Request timeout")

        except Exception as e:
            results[endpoint] = {"status": "error", "error": str(e)}
            print(f"  ✗ Error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    successful = len(
        [
            r
            for r in results.values()
            if r.get("status") == "success" and r.get("success_flag")
        ]
    )
    total = len(endpoints)

    print(f"Successful endpoints: {successful}/{total}")
    print(f"Success rate: {(successful/total)*100:.1f}%")

    if successful == total:
        print("\n✓ All AI dashboard endpoints are working correctly!")
    elif successful > 0:
        print(f"\n⚠ {total-successful} endpoints need attention")
    else:
        print("\n✗ Dashboard server may not be running or endpoints have issues")

    # Detailed results
    print(f"\nDetailed Results:")
    for endpoint, result in results.items():
        status_icon = (
            "✓"
            if result.get("status") == "success" and result.get("success_flag")
            else "✗"
        )
        print(f"{status_icon} {endpoint}: {result.get('status', 'unknown')}")

    return results


def test_dashboard_page():
    """Test that the main dashboard page loads"""

    base_url = "http://localhost:5000"

    print(f"\nTesting dashboard page access...")

    try:
        response = requests.get(f"{base_url}/", timeout=10)

        if response.status_code == 200:
            print(f"  ✓ Dashboard page loads successfully")
            return True
        else:
            print(f"  ✗ Dashboard page returned HTTP {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"  ✗ Could not connect to dashboard server")
        print(f"    Make sure to run: python dashboard/app.py")
        return False

    except Exception as e:
        print(f"  ✗ Error accessing dashboard: {e}")
        return False


def main():
    """Main test function"""

    print("AI Dashboard Integration Test")
    print(f"Timestamp: {datetime.now()}")
    print("")

    # Test dashboard page
    page_ok = test_dashboard_page()

    if page_ok:
        # Test API endpoints
        results = test_dashboard_endpoints()

        # Save results
        results_file = "logs/ai_dashboard_test_results.json"
        os.makedirs("logs", exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(
                {"timestamp": datetime.now().isoformat(), "results": results},
                f,
                indent=2,
            )

        print(f"\nResults saved to: {results_file}")

    else:
        print("\nSkipping API tests - dashboard not accessible")

    print(f"\nTo start the dashboard server, run:")
    print(f"  cd dashboard")
    print(f"  python app.py")
    print(f"\nThen visit: http://localhost:5000")


if __name__ == "__main__":
    main()
