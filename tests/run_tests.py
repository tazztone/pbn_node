#!/usr/bin/env python
import os
import subprocess
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"Running tests from: {script_dir}")

    args = [sys.executable, "-m", "pytest"]

    # Parse our custom flags and separate them from pytest args
    pytest_args = []
    run_quality = False

    for arg in sys.argv[1:]:
        if arg == "--quality":
            run_quality = True
        elif arg == "--all":
            run_quality = True
        else:
            pytest_args.append(arg)

    if pytest_args:
        args.extend(pytest_args)
    else:
        # Default run: only fast unit and integration tests
        args.extend(["unit", "integration"])
        if run_quality:
            args.append("quality")

    print(f"Executing: {' '.join(args)}")

    result = subprocess.run(args, cwd=script_dir)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
