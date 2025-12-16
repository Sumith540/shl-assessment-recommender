from playwright.sync_api import sync_playwright
import pandas as pd
import json
import time

CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

def main():
    products = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        def handle_response(response):
            try:
                if "product" in response.url.lower() and response.request.resource_type == "xhr":
                    data = response.json()
                    if isinstance(data, dict) and "items" in data:
                        for item in data["items"]:
                            products.append({
                                "name": item.get("name"),
                                "url": "https://www.shl.com" + item.get("url", ""),
                                "description": item.get("description", ""),
                                "duration": item.get("duration", ""),
                                "test_type": item.get("testType", []),
                                "adaptive_support": "Yes" if item.get("adaptive") else "No",
                                "remote_support": "Yes" if item.get("remote") else "No"
                            })
            except Exception:
                pass

        page.on("response", handle_response)

        print("Opening catalog page...")
        page.goto(CATALOG_URL, timeout=60000)

        # Allow time
        time.sleep(10)

        browser.close()

    df = pd.DataFrame(products).drop_duplicates(subset=["url"])
    df.to_csv("data/shl_assessments.csv", index=False)

    print(f"\nSaved {len(df)} assessments to data/shl_assessments.csv")

if __name__ == "__main__":
    main()
