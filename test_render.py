import sys
# Just ensuring we can import
try:
    from pbn_renderer import PBNRenderer
    print("Import OK")
except Exception as e:
    print(f"Error: {e}")
