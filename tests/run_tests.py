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
    run_quality_only = False
    run_all = False

    for arg in sys.argv[1:]:
        if arg == "--quality":
            run_quality_only = True
        elif arg == "--all":
            run_all = True
        else:
            pytest_args.append(arg)

    if pytest_args:
        args.extend(pytest_args)
    else:
        if run_all:
            args.extend(["unit", "integration", "quality"])
        elif run_quality_only:
            args.extend(["quality"])
        else:
            # Default: only run fast unit and integration tests
            args.extend(["unit", "integration"])

    print(f"Executing: {' '.join(args)}")

    result = subprocess.run(args, cwd=script_dir)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
