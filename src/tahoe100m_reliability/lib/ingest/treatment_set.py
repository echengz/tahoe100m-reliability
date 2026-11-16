"""
Defines a treatment set.
"""
import csv
import json
import logging

from pathlib import Path
from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class TreatmentSet(BaseModel):
    """
    Class to represent a treatment (drug and dosage) along with the
    corresponding samples (replicate treatments) and the parquets that
    contain cell data for this treatment.
    """
    drug: str
    dosage: float
    dosage_unit: str

    samples: set[str] = set({})
    parquet_files: set[str] = set({})

    # mapped from samples
    plates: dict[str, str] = {}

    def add_sample(self, sample: str) -> None:
        """Add an associated sample."""
        self.samples.add(sample)

    def add_parquet_files(self, parquet_files: set[str]) -> None:
        """Add associated parquet files"""
        self.parquet_files.update(parquet_files)

    @classmethod
    def load_list_from_file(
        cls,
        treatments_file: Path,
        drug_dosages: Path | None = None
    ) -> list["TreatmentSet"]:
        """
        Load list of TreatmentSet from file. If drug dosages are
        provided, subsets to that list of drugs.
        """
        keyed_drug_dosages = None
        if drug_dosages:
            log.info("Drug dosage CSV provided, will subset treatments to this list")
            keyed_drug_dosages = set({})
            with drug_dosages.open('r') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    keyed_drug_dosages.add(f'{row['drug'].strip()}_{row['dosage'].strip()}')

        treatment_sets = []
        with treatments_file.open('r') as fh:
            for x in json.load(fh):
                if keyed_drug_dosages is not None:
                    key = f'{x['drug'].strip()}_{x['dosage']}'
                    if key not in keyed_drug_dosages:
                        continue

                treatment_sets.append(cls(**x))

        log.info(f"Loaded {len(treatment_sets)} treatment sets")
        return treatment_sets

    @classmethod
    def load_clusters_from_file(
        cls,
        treatments_file: Path,
        drug_dosages: Path, 
        target_clusters: list[int] | None
    ) -> dict[int, "TreatmentSet"]:
        """
        Load clusters of treatments from files.
        """
        keyed_drug_dosages = {}
        with drug_dosages.open('r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = f'{row['drug'].strip()}_{row['dosage'].strip()}'
                if target_clusters and int(row['cluster']) not in target_clusters:
                    continue

                keyed_drug_dosages[key] = int(row['cluster'])

        clustered_treatment_sets = {}
        with treatments_file.open('r') as fh:
            for x in json.load(fh):
                key = f'{x['drug'].strip()}_{x['dosage']}'
                if key not in keyed_drug_dosages:
                    continue

                cluster = keyed_drug_dosages[key]
                if cluster not in clustered_treatment_sets:
                    clustered_treatment_sets[cluster] = []

                clustered_treatment_sets[cluster].append(cls(**x))

        log.info(f"Loaded {len(clustered_treatment_sets)} clusters of treatment sets")
        for cluster, treatments in clustered_treatment_sets.items():
            log.info(f"Cluster {cluster}: {len(treatments)} treatments")

        return clustered_treatment_sets

