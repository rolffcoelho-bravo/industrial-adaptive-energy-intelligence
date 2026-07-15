from pathlib import Path

import pandas as pd
import pytest

from iaei.contracts import load_yaml
from iaei.visualization.charts import plot_industrial_load_profile


def test_visualization_contract_defines_five_brief_figures() -> None:
    contract = load_yaml(Path("configs/visualization_contract.yml"))["visualization"]
    assert len(contract["brief_figures"]) == 5
    assert contract["background"] == "light"
    assert contract["publication_gate"]["exact_brief_figures"] == 5


def test_visualization_rejects_empty_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        plot_industrial_load_profile(
            pd.DataFrame(columns=["timestamp", "energy_demand"]),
            tmp_path / "chart.png",
            source="UCI",
            sample="not applicable",
        )
