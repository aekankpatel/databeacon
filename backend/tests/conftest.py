import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    os.environ["DATABEACON_DATA_DIR"] = str(d)
    from databeacon.config import reset_settings_cache

    reset_settings_cache()
    return d


@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory):
    from sample_data.generate_ecommerce import generate

    p = tmp_path_factory.mktemp("csv") / "ecommerce_orders.csv"
    generate().to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def registry(data_dir):
    from databeacon.config import get_settings
    from databeacon.datasets import DatasetRegistry

    return DatasetRegistry(get_settings().datasets_dir)


@pytest.fixture(scope="session")
def dataset(registry, sample_csv):
    return registry.ingest_file(sample_csv, name="E-commerce orders")
