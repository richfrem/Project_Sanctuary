import json
from pathlib import Path
from council_orchestrator.orchestrator.packets.schema import *
from council_orchestrator.orchestrator.packets.emitter import emit_packet

def test_emitter_writes_one_line(tmp_path: Path):
    pkt = CouncilRoundPacket(
        timestamp="2025-01-01T00:00:00Z",
        session_id="run_X",
        round_id=1,
        member_id="auditor",
        engine="ollama",
        seed=7,
        prompt_hash="def456",
        inputs={},
        decision="review",
        rationale="...",
        confidence=0.66,
        citations=[],
        rag={},
        cag={},
        cost={},
        errors=[]
    )
    out = tmp_path
    emit_packet(pkt, jsonl_dir=str(out), stream_stdout=False, schema_path=None)
    f = (out / "run_X" / "round_1.jsonl")
    assert f.exists()
    lines = f.read_text().strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["memory_directive"]["tier"]
    assert "retrieval" in obj and "novelty" in obj and "conflict" in obj