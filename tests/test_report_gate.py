from pathlib import Path

import pytest

from iaei.contracts import ContractError, validate_report_payload


def test_template_cannot_be_published() -> None:
    template = Path("examples/report_payload.template.json")
    with pytest.raises(ContractError):
        validate_report_payload(template)
