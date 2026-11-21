"""Install requirements.txt using the Python running this file."""

from pathlib import Path
import subprocess
import sys


def main():
    print("Python:", sys.version.split()[0], flush=True)
    # Both pinned numerical packages have wheels for Python 3.11-3.13.
    if sys.version_info[:2] not in [(3, 11), (3, 12), (3, 13)]:
        print("This requirements.txt targets Python 3.11-3.13; 3.12 has been tested.")
        print("Select one of those interpreters and run this file again.")
        return 1

    requirements = Path(__file__).resolve().with_name("requirements.txt")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", str(requirements)
    ])
    if result.returncode == 0:
        print("Installation completed. You can now run wine_quality.py.")
    else:
        print("Installation failed. Check pip's error above and your network/permissions.")
    return result.returncode


if __name__ == "__main__":
    exit_code = main()
    if sys.stdin.isatty():
        input("Press Enter to close...")
    sys.exit(exit_code)
