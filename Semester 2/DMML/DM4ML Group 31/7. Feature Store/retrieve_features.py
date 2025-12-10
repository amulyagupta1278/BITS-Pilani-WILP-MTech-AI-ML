
import os, pandas as pd
from custom_feature_store import CustomFeatureStore

store = CustomFeatureStore(repo_dir="/content/drive/MyDrive/DMML_Project/data/feature_store", entity_key="row_id", snapshot_prefix="generic_features")

print("=== Feature Metadata ===")
print(store.metadata().to_string(index=False))

snap = pd.read_csv("/content/drive/MyDrive/DMML_Project/data/feature_store/feature_data/generic_features_2025-08-22_v1.csv")
ids = snap['row_id'].astype(str).head(3).tolist()

features = store.get_features(entity_ids=ids, as_of="2025-08-22", columns=['gender', 'SeniorCitizen', 'tenure', 'PhoneService', 'services_count'])

print("\n=== Retrieved Features ===")
print(features.to_string(index=False))

out_path = "/content/drive/MyDrive/DMML_Project/data/feature_store/reports/retrieved_features_sample.csv"
features.to_csv(out_path, index=False)
print(f"\nSaved sample retrieval to {out_path}")
