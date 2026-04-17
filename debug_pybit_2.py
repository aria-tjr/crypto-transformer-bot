from pybit.unified_trading import HTTP
try:
    print("Attempting with demo=True...")
    session = HTTP(demo=True)
    print(f"Success! Endpoint: {session.endpoint}")
except Exception as e:
    print(f"Failed with demo=True: {e}")

try:
    print("Attempting with domain='demo'...")
    session = HTTP(domain="demo")
    print(f"Success! Endpoint: {session.endpoint}")
except Exception as e:
    print(f"Failed with domain='demo': {e}")
