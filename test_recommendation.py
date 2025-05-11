"""
Test script for the recommendation API.
This checks if the recommendation endpoint is working properly.
"""
import requests
import json
import argparse
import sys

def test_recommendation_api(customer_id, base_url="http://localhost:8000"):
    """Test the recommendation API with the provided customer ID."""
    print(f"Testing recommendation API for customer {customer_id}")
    print(f"Base URL: {base_url}")
    
    # Test the API endpoint
    try:
        url = f"{base_url}/recommend/{customer_id}"
        print(f"Requesting: {url}")
        
        response = requests.get(url, timeout=30)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Retrieved {len(data)} products")
            # Print the first 3 products
            for i, product in enumerate(data[:3]):
                print(f"\nProduct {i+1}:")
                print(f"  Name: {product.get('name', 'N/A')}")
                print(f"  Category: {product.get('category', 'N/A')}")
                print(f"  Description: {product.get('description', 'N/A')[:50]}...")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

def test_dependency_services():
    """Check if all required services are running."""
    services = [
        {"name": "Product API", "url": "http://192.168.43.232:3000/product"},
        {"name": "Orders API", "url": "http://192.168.43.232:3000/order"},
        {"name": "Lands API", "url": "http://192.168.43.232:3000/lands/all"},
        {"name": "Ollama LLM", "url": "http://192.168.43.232:11434/api/version"}
    ]
    
    print("Checking dependency services:")
    
    all_services_ok = True
    for service in services:
        try:
            print(f"Testing {service['name']}...", end="")
            response = requests.get(service["url"], timeout=5)
            if response.status_code == 200:
                print(" OK")
            else:
                print(f" ERROR (Status: {response.status_code})")
                all_services_ok = False
        except requests.RequestException as e:
            print(f" FAILED ({str(e)})")
            all_services_ok = False
    
    return all_services_ok

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the recommendation API.")
    parser.add_argument("customer_id", help="Customer ID to test with")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API server")
    parser.add_argument("--skip-dependency-check", action="store_true", help="Skip dependency service checks")
    
    args = parser.parse_args()
    
    if not args.skip_dependency_check:
        print("\n=== CHECKING DEPENDENCY SERVICES ===")
        services_ok = test_dependency_services()
        if not services_ok:
            print("\nWARNING: Some dependency services are not available.")
            print("The recommendation API may not work correctly.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    print("\n=== TESTING RECOMMENDATION API ===")
    success = test_recommendation_api(args.customer_id, args.url)
    
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")
        sys.exit(1) 