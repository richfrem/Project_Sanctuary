from council_orchestrator.orchestrator.packets.schema import CouncilRoundPacket, RetrievalField, NoveltyField, ConflictField, MemoryDirectiveField

def test_packet_phase2_fields_exist_and_types():
    p = CouncilRoundPacket(
        timestamp="2025-01-01T00:00:00Z",
        session_id="s",
        round_id=1,
        member_id="coordinator",
        engine="gemini",
        seed=1,
        prompt_hash="abc123",
        inputs={},
        decision="approve",
        rationale="ok",
        confidence=0.8,
        citations=[],
        rag={},
        cag={},
        cost={},
        errors=[]
    )
    assert hasattr(p, "retrieval") and isinstance(p.retrieval, RetrievalField)
    assert hasattr(p, "novelty") and isinstance(p.novelty, NoveltyField)
    assert hasattr(p, "conflict") and isinstance(p.conflict, ConflictField)
    assert hasattr(p, "memory_directive") and isinstance(p.memory_directive, MemoryDirectiveField)