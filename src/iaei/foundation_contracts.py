from __future__ import annotations

from typing import Any

from iaei.contracts import (
    ContractError,
    _validate_payload,
    load_json,
    load_yaml,
    validate_optimization_governance,
    validate_v2_architecture_contract,
)
from iaei.paths import CONFIGS, ROOT, SCHEMAS


def validate_foundation_model_contract() -> dict[str, Any]:
    contract = load_yaml(CONFIGS / "foundation_model_contract.yml")
    schema = load_json(SCHEMAS / "foundation_model_contract.schema.json")
    _validate_payload(
        contract,
        schema,
        label="Gate 6D foundation-model benchmark contract",
    )

    architecture = validate_v2_architecture_contract()
    optimization = validate_optimization_governance()
    if architecture["v1_baseline"]["locked_test_access_permitted"] is not False:
        raise ContractError("Gate 6D cannot weaken the V1 locked-test boundary")
    if optimization["evidence_boundary"][
        "locked_test_partition_permitted"
    ] is not False:
        raise ContractError("Gate 6D cannot admit the locked-test partition")
    if contract["promotion"]["automatic_promotion_permitted"] is not False:
        raise ContractError("Gate 6D cannot permit automatic promotion")

    candidate_ids: set[str] = set()
    weight_hashes: set[str] = set()
    context_length = int(contract["benchmark_protocol"]["context_length_intervals"])
    for candidate in contract["candidate_models"]:
        candidate_id = str(candidate["candidate_id"])
        weight_hash = str(candidate["weight_sha256"])
        if candidate_id in candidate_ids:
            raise ContractError(f"Duplicate Gate 6D candidate: {candidate_id}")
        if weight_hash in weight_hashes:
            raise ContractError(f"Duplicate Gate 6D weight identity: {weight_hash}")
        candidate_ids.add(candidate_id)
        weight_hashes.add(weight_hash)

        if int(candidate["maximum_supported_context_intervals"]) < context_length:
            raise ContractError(
                f"Gate 6D candidate cannot support frozen context: {candidate_id}"
            )
        if candidate["promotion_eligible"] and not candidate["commercial_use_eligible"]:
            raise ContractError(
                f"Non-commercial Gate 6D candidate is promotion eligible: {candidate_id}"
            )

    controls = contract["network_and_artifact_controls"]
    prohibited_true = (
        "gate_6d1_network_access_permitted",
        "gate_6d1_model_download_permitted",
        "gate_6d1_inference_permitted",
        "unpinned_revision_permitted",
        "remote_code_trust_permitted",
        "arbitrary_model_code_permitted",
    )
    if any(bool(controls[field]) for field in prohibited_true):
        raise ContractError("Gate 6D1 weakens an implementation-only control")
    return contract


def validate_gate_6d1_closure_manifest() -> dict[str, Any]:
    manifest = load_json(ROOT / "outputs" / "v2" / "gate_6d1_closure_manifest.json")
    schema = load_json(SCHEMAS / "gate_6d1_closure_manifest.schema.json")
    _validate_payload(manifest, schema, label="Gate 6D1 closure manifest")

    gate_6c_closure = load_json(
        ROOT / "outputs" / "v2" / "gate_6c" / "gate_6c_closure_manifest.json"
    )
    if gate_6c_closure.get("status") != "closed":
        raise ContractError("Gate 6D1 requires the closed Gate 6C predecessor")
    if gate_6c_closure.get("next_gate") != "6D":
        raise ContractError("Gate 6C closure does not authorize the Gate 6D sequence")
    return manifest
