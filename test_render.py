# Just ensuring we can import
try:
    from pbn_renderer import PBNRenderer  # noqa: F401

    print("Import OK")
except Exception as e:
    print(f"Error: {e}")
