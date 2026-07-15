from pathlib import Path

from iaei.reporting import build_technical_brief

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "outputs" / "report_payload.json"
OUTPUT = ROOT / "outputs" / "brief" / "industrial_adaptive_energy_intelligence_technical_brief.pdf"

if __name__ == "__main__":
    result = build_technical_brief(PAYLOAD, OUTPUT)
    print(result)
