import requests

proxy = {
    "http": "http://143.198.42.182:31280",
    "https": "http://143.198.42.182:31280",
}

try:
    response = requests.get("https://www.google.com", proxies=proxy, timeout=5)
    print("✅ Proxy is working!")
except Exception as e:
    print("❌ Proxy failed:", e)
