"""``python -m spire.design_db …`` — same CLI as the ``spire`` console script."""
import sys

from spire.design_db.cli import main

if __name__ == "__main__":
    sys.exit(main(["db", *sys.argv[1:]]))
