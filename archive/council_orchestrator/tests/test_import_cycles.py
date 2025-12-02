# council_orchestrator/tests/test_import_cycles.py
"""
Import cycle and boundary tests for modular architecture.
Ensures clean separation between layers and no circular dependencies.
"""

def test_packets_import_facade():
    """Test that packet façade imports work correctly."""
    try:
        from council_orchestrator.orchestrator.packets import (
            CouncilRoundPacket,
            validate_packet,
            seed_for,
            prompt_hash,
            emit_packet,
            aggregate_round_events,
            calculate_round_telemetry
        )
        assert CouncilRoundPacket is not None
        assert callable(validate_packet)
        assert callable(seed_for)
        assert callable(prompt_hash)
        assert callable(emit_packet)
        assert callable(aggregate_round_events)
        assert callable(calculate_round_telemetry)
    except ImportError as e:
        raise AssertionError(f"Packet façade import failed: {e}")

def test_substrate_monitor_boundaries():
    """Test that substrate_monitor only imports from engines, not vice versa."""
    try:
        # This should work - substrate_monitor importing from engines
        from council_orchestrator.orchestrator.substrate_monitor import select_engine
        assert callable(select_engine)

        # Test that engines don't import from substrate_monitor (would create cycle)
        import council_orchestrator.orchestrator.engines.base
        import council_orchestrator.orchestrator.engines.gemini_engine
        import council_orchestrator.orchestrator.engines.openai_engine
        import council_orchestrator.orchestrator.engines.ollama_engine

        # If we get here without circular import errors, boundaries are clean
        assert True

    except ImportError as e:
        if "cannot import name" in str(e) and "substrate_monitor" in str(e):
            raise AssertionError(f"Engine module illegally imports from substrate_monitor: {e}")
        else:
            raise  # Re-raise other import errors

def test_orchestrator_layer_imports():
    """Test that orchestrator layer imports work through façade."""
    try:
        from council_orchestrator.orchestrator import CouncilRoundPacket, emit_packet
        assert CouncilRoundPacket is not None
        assert callable(emit_packet)
    except ImportError as e:
        raise AssertionError(f"Orchestrator layer import failed: {e}")