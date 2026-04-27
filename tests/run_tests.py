#!/usr/bin/env python
import os
import sys
import subprocess

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"Running tests from: {script_dir}")

    args = [sys.executable, "-m", "pytest"]

    if len(sys.argv) > 1:
        args.extend(sys.argv[1:])
    else:
        args.extend(["unit", "integration"])

    print(f"Executing: {' '.join(args)}")

    result = subprocess.run(args, cwd=script_dir)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
