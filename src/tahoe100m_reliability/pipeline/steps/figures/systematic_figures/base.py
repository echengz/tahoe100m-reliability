"""
Base class for systematic figures.
"""
import re
import logging

import pandas as pd

from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class BaseSystematicFigures:
    """Base class for generating systematic figures."""

    def __init__(self, systema_dir: Path, output_dir: Path) -> None:
        """Initializer."""
        self.systema_dir = systema_dir
        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def grab_latest_systema_results(self, prefix: str) -> pd.DataFrame:
        """Grab the latest systema results."""
        prefix_path = Path(prefix)
        search_dir = self.systema_dir / prefix_path.parent
        filename_prefix = prefix_path.name

        dated_pattern = re.compile(
            rf"^{re.escape(filename_prefix)}_(\d{{8}}-\d{{6}})\.csv$"
        )

        latest_file = {"path": None, "timestamp": None}

        for path in search_dir.glob(f"{filename_prefix}_*.csv"):
            match = dated_pattern.match(path.name)

            if match:
                timestamp = datetime.strptime(match.group(1), "%Y%m%d-%H%M%S")

                if latest_file["timestamp"] is None or timestamp > latest_file["timestamp"]:
                    latest_file = {"path": path, "timestamp": timestamp}

        if latest_file["path"] is None:
            raise FileNotFoundError(f"Cannot find {prefix} file in {search_dir} (may be missing timestamp?)")

        log.info("Loading latest results from %s", latest_file["path"])
        return pd.read_csv(latest_file["path"])
