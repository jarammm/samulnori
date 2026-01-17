import os
import shutil
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

import argbind


# Allow local imports
@contextmanager
def chdir(path: str):
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_path))

with chdir(_path):
    from tria.data.preprocess import create_manifests
    from tria.constants import DATA_DIR, MANIFESTS_DIR
    from tria.util import get_info

warnings.filterwarnings("ignore", category=UserWarning)

################################################################################
# Create CSV manifests for base data
################################################################################


@argbind.bind(without_prefix=True)
def main(seed: int = 0):

    metadata_fns = {
        "drums": lambda p: str(p),
        "sample_rate": lambda p: get_info(str(p))[1],
        "duration": lambda p: get_info(str(p))[0],
    }

    ########################################
    # GUGAK
    ########################################


    print("Creating manifests for gugak")

    out_paths = create_manifests(
        DATA_DIR / "gugak",
        ext=".wav",
        output_dir=MANIFESTS_DIR / "gugak",
        split=(0.8, 0.1, 0.1),
        attributes=metadata_fns,
        seed=seed,
    )
    print(
        "Created manifests at ",
        f"{[str(p.relative_to(MANIFESTS_DIR)) for p in out_paths.values()]}",
    )


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
