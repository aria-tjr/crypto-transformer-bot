from pybit.unified_trading import HTTP
try:
    session = HTTP(testnet=False)
    print(f"Attributes: {dir(session)}")
    # Look for url related attributes
    for attr in dir(session):
        if 'url' in attr.lower() or 'endpoint' in attr.lower():
            print(f"{attr}: {getattr(session, attr)}")
except Exception as e:
    print(e)
