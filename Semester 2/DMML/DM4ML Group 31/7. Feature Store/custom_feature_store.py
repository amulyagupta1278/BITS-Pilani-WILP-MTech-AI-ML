
import os, pandas as pd
from typing import List, Optional

class CustomFeatureStore:
    def __init__(self, repo_dir: str = "/content/drive/MyDrive/DMML_Project/data/feature_store", entity_key: str = "row_id", snapshot_prefix: str = "generic_features"):
        self.repo_dir = repo_dir
        self.entity_key = entity_key
        self.snapshot_prefix = snapshot_prefix
        self.meta_path = os.path.join(repo_dir, "feature_metadata.csv")
        self.data_dir = os.path.join(repo_dir, "feature_data")
        os.makedirs(self.data_dir, exist_ok=True)

    def metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.meta_path)

    def _latest_snapshot_path(self) -> str:
        files = [f for f in os.listdir(self.data_dir) if f.startswith(self.snapshot_prefix)]
        if not files:
            raise FileNotFoundError("No feature snapshots found.")
        return os.path.join(self.data_dir, sorted(files)[-1])

    def _snapshot_for_date(self, as_of: str) -> str:
        return os.path.join(self.data_dir, f"{self.snapshot_prefix}_{as_of}_v1.csv")

    def get_features(self, entity_ids: Optional[List[str]] = None, as_of: Optional[str] = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        if as_of:
            path = self._snapshot_for_date(as_of)
            if not os.path.exists(path):
                path = self._latest_snapshot_path()
        else:
            path = self._latest_snapshot_path()

        df = pd.read_csv(path)
        if entity_ids is not None:
            ids = set(map(str, entity_ids))
            df = df[df[self.entity_key].astype(str).isin(ids)]
        if columns is not None:
            keep = [self.entity_key] + [c for c in columns if c in df.columns]
            df = df[keep]
        return df.reset_index(drop=True)
