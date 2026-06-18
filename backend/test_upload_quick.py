import requests

# Create a tiny test python file in memory
files = {'file': ('hello.py', b'def hello():\n    return "world"\n', 'text/plain')}

print("Sending upload request (this may take a while on first run - model download)...")
try:
    r = requests.post('http://localhost:8000/api/upload', files=files, timeout=300)
    print(f"Status: {r.status_code}")
    print(f"Response: {r.json()}")
except Exception as e:
    print(f"Error: {e}")
