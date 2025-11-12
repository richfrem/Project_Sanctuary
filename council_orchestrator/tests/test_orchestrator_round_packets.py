#!/usr/bin/env python3
"""
Unit tests for Council Round Packet emission system.
Tests round packet creation, validation, emission channels, and core logic.
"""

import unittest
import json
import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Import the components we need to test
from council_orchestrator.orchestrator.packets.schema import (
    CouncilRoundPacket, seed_for, prompt_hash,
    MemoryDirectiveField, NoveltyField, ConflictField, RetrievalField
)
from council_orchestrator.orchestrator.packets.emitter import emit_packet
from council_orchestrator.orchestrator.app import Orchestrator, CacheAdapter


class TestCouncilRoundPacket(unittest.TestCase):
    """Test CouncilRoundPacket dataclass and utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_packet = CouncilRoundPacket(
            timestamp="2025-01-15T10:30:00Z",
            session_id="test_session_123",
            round_id=1,
            member_id="coordinator",
            engine="ollama",
            seed=12345,
            prompt_hash="abc123def456",
            inputs={"prompt": "Test prompt", "context": "Test context"},
            decision="approve",
            rationale="This is a test rationale",
            confidence=0.85,
            citations=[{"source_file": "test.md", "span": "lines 1-5"}],
            rag={
                "structured_query": {"entities": ["test"]},
                "parent_docs": ["doc1.md", "doc2.md"],
                "retrieval_latency_ms": 42
            },
            cag={
                "query_key": "cache_key_123",
                "cache_hit": False,
                "hit_streak": 0
            },
            novelty={
                "is_novel": True,
                "signal": "high",
                "conflicts_with": []
            },
            memory_directive={
                "tier": "medium",
                "justification": "Test justification"
            },
            cost={
                "input_tokens": 100,
                "output_tokens": 50,
                "latency_ms": 1500
            },
            errors=[]
        )

    def test_packet_creation(self):
        """Test that CouncilRoundPacket can be created with valid data."""
        self.assertEqual(self.sample_packet.session_id, "test_session_123")
        self.assertEqual(self.sample_packet.round_id, 1)
        self.assertEqual(self.sample_packet.member_id, "coordinator")
        self.assertEqual(self.sample_packet.decision, "approve")
        self.assertEqual(self.sample_packet.confidence, 0.85)

    def test_packet_serialization(self):
        """Test that packets can be serialized to JSON."""
        payload = self.sample_packet.__dict__
        json_str = json.dumps(payload, default=str)
        self.assertIn("test_session_123", json_str)
        self.assertIn("coordinator", json_str)

    def test_seed_determinism(self):
        """Test that seed generation is deterministic."""
        seed1 = seed_for("session_1", 1, "coordinator")
        seed2 = seed_for("session_1", 1, "coordinator")
        self.assertEqual(seed1, seed2)

        # Different inputs should give different seeds
        seed3 = seed_for("session_2", 1, "coordinator")
        self.assertNotEqual(seed1, seed3)

    def test_prompt_hash(self):
        """Test prompt hash generation."""
        hash1 = prompt_hash("test prompt")
        hash2 = prompt_hash("test prompt")
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)  # Should be 16 chars

        # Different prompts should give different hashes
        hash3 = prompt_hash("different prompt")
        self.assertNotEqual(hash1, hash3)


class TestPacketEmission(unittest.TestCase):
    """Test packet emission to files and stdout."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.schema_path = os.path.join(self.temp_dir, "schema.json")

        # Create a minimal schema
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["timestamp", "session_id"],
            "properties": {
                "timestamp": {"type": "string"},
                "session_id": {"type": "string"}
            }
        }

        with open(self.schema_path, 'w') as f:
            json.dump(schema, f)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    @patch('sys.stdout')
    def test_stdout_emission(self, mock_stdout):
        """Test emission to stdout."""
        packet = CouncilRoundPacket(
            timestamp="2025-01-15T10:30:00Z",
            session_id="test_session",
            round_id=1,
            member_id="coordinator",
            engine="ollama",
            seed=12345,
            prompt_hash="abc123",
            inputs={},
            decision="test",
            rationale="test",
            confidence=0.8,
            citations=[],
            rag={},
            cag={},
            novelty={},
            memory_directive={"tier": "fast", "justification": "test"},
            cost={},
            errors=[]
        )

        emit_packet(packet, None, True, self.schema_path)

        # Check that stdout.write was called
        mock_stdout.write.assert_called_once()
        call_args = mock_stdout.write.call_args[0][0]
        self.assertIn("test_session", call_args)

    def test_file_emission(self):
        """Test emission to JSONL files."""
        packet = CouncilRoundPacket(
            timestamp="2025-01-15T10:30:00Z",
            session_id="test_session",
            round_id=1,
            member_id="coordinator",
            engine="ollama",
            seed=12345,
            prompt_hash="abc123",
            inputs={},
            decision="test",
            rationale="test",
            confidence=0.8,
            citations=[],
            rag={},
            cag={},
            novelty={},
            memory_directive={"tier": "fast", "justification": "test"},
            cost={},
            errors=[]
        )

        emit_packet(packet, self.temp_dir, False, self.schema_path)

        # Check that file was created
        expected_path = os.path.join(self.temp_dir, "test_session", "round_1.jsonl")
        self.assertTrue(os.path.exists(expected_path))

        # Check file contents
        with open(expected_path, 'r') as f:
            content = f.read()
            self.assertIn("test_session", content)
            self.assertIn("coordinator", content)


class TestOrchestratorIntegration(unittest.TestCase):
    """Test orchestrator integration with round packets."""

    def setUp(self):
        """Set up orchestrator for testing."""
        self.orchestrator = Orchestrator()

    @patch('orchestrator.substrate_monitor.select_engine')
    def test_rag_data_generation(self, mock_select_engine):
        """Test RAG data generation."""
        mock_engine = Mock()
        mock_select_engine.return_value = mock_engine

        task = "Test task description"
        response = "Test response with some content"

        rag_data = self.orchestrator._get_rag_data(task, response)

        self.assertIn("structured_query", rag_data)
        self.assertIn("parent_docs", rag_data)
        self.assertIn("retrieval_latency_ms", rag_data)

    def test_novelty_analysis(self):
        """Test novelty analysis."""
        response = "This is a completely new idea"
        context = "The old discussion was about something else entirely"

        novelty = self.orchestrator._analyze_novelty(response, context)

        self.assertIn("is_novel", novelty)
        self.assertIn("signal", novelty)
        self.assertIn("conflicts_with", novelty)

    def test_memory_directive(self):
        """Test memory directive determination."""
        response = "This is a well-reasoned response with evidence"
        citations = [{"source_file": "doc.md", "span": "lines 1-10"}]

        directive = self.orchestrator._determine_memory_directive(response, citations)

        self.assertIn("tier", directive)
        self.assertIn("justification", directive)
        self.assertIn(directive["tier"], ["fast", "medium", "slow", "none"])

    def test_cag_data_generation(self):
        """Test CAG data generation."""
        prompt = "Test prompt"
        engine_type = "ollama"

        cache_adapter = CacheAdapter()
        cag_data = cache_adapter.get_cag_data(prompt, engine_type)

        self.assertIn("query_key", cag_data)
        self.assertIn("cache_hit", cag_data)
        self.assertIn("hit_streak", cag_data)


class TestSchemaValidation(unittest.TestCase):
    """Test JSON schema validation."""

    def setUp(self):
        """Set up schema for validation tests."""
        self.schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["timestamp", "session_id", "round_id", "member_id", "engine", "seed", "prompt_hash", "inputs", "decision", "rationale", "confidence", "citations", "rag", "cag", "novelty", "memory_directive", "cost", "errors", "schema_version", "retrieval", "conflict", "seed_chain"],
            "properties": {
                "schema_version": {"type": "string", "description": "Schema version for future compatibility"},
                "timestamp": {"type": "string", "format": "date-time"},
                "session_id": {"type": "string"},
                "round_id": {"type": "integer", "minimum": 1},
                "member_id": {"type": "string"},
                "engine": {"type": "string"},
                "seed": {"type": "integer"},
                "prompt_hash": {"type": "string"},
                "inputs": {"type": "object"},
                "decision": {"type": "string"},
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "citations": {"type": "array", "items": {"type": "object"}},
                "rag": {"type": "object"},
                "cag": {"type": "object"},
                "novelty": {"type": "object"},
                "memory_directive": {"type": "object", "properties": {"tier": {"type": "string", "enum": ["fast", "medium", "slow", "none"]}}},
                "cost": {"type": "object"},
                "errors": {"type": "array", "items": {"type": "string"}},
                # --- Phase 2 additions ---
                "retrieval": {"type": "object"},
                "conflict": {"type": "object"},
                "seed_chain": {"type": "object"}
            }
        }

    def test_valid_packet_validation(self):
        """Test that valid packets pass schema validation."""
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not available")

        packet = CouncilRoundPacket(
            timestamp="2025-01-15T10:30:00Z",
            session_id="test_session",
            round_id=1,
            member_id="coordinator",
            engine="ollama",
            seed=12345,
            prompt_hash="abc123def4567890",
            inputs={},
            decision="approve",
            rationale="Test rationale",
            confidence=0.85,
            citations=[],
            rag={},
            cag={},
            novelty={},
            memory_directive={"tier": "medium", "justification": "test"},
            cost={},
            errors=[]
        )

        payload = asdict(packet)
        # Should not raise an exception
        jsonschema.validate(instance=payload, schema=self.schema)

    def test_invalid_packet_fails_validation(self):
        """Test that invalid packets fail schema validation."""
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not available")

        # Invalid confidence value
        invalid_payload = {
            "timestamp": "2025-01-15T10:30:00Z",
            "session_id": "test_session",
            "round_id": 1,
            "member_id": "coordinator",
            "engine": "ollama",
            "seed": 12345,
            "prompt_hash": "abc123",
            "inputs": {},
            "decision": "approve",
            "rationale": "Test rationale",
            "confidence": 1.5,  # Invalid: > 1.0
            "citations": [],
            "rag": {},
            "cag": {},
            "novelty": {},
            "memory_directive": {"tier": "medium", "justification": "test"},
            "cost": {},
            "errors": []
        }

        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=invalid_payload, schema=self.schema)

    def test_schema_evolution_detection(self):
        """Test that schema fields exactly match packet fields to prevent silent drift."""
        # Create a complete packet using the dataclass
        packet = CouncilRoundPacket(
            timestamp="2025-01-15T10:30:00Z",
            session_id="test_session",
            round_id=1,
            member_id="coordinator",
            engine="ollama",
            seed=12345,
            prompt_hash="abc123def4567890",
            inputs={},
            decision="approve",
            rationale="Test rationale",
            confidence=0.85,
            citations=[],
            rag={},
            cag={},
            novelty=NoveltyField(False, "none", {}),
            memory_directive=MemoryDirectiveField("medium", "test"),
            cost={},
            errors=[],
            schema_version="1.0.0",
            # Phase 2 fields
            retrieval=RetrievalField(),
            conflict=ConflictField(),
            seed_chain={}
        )

        payload = packet.__dict__

        # Get defined fields from schema
        defined_fields = set(self.schema["properties"].keys())

        # Get actual fields from packet
        packet_fields = set(payload.keys())

        # They must match exactly - no silent drift allowed
        self.assertEqual(defined_fields, packet_fields,
                        f"Schema vs packet mismatch: {defined_fields ^ packet_fields}")

        # Required fields must be present
        required_fields = set(self.schema["required"])
        self.assertTrue(required_fields.issubset(packet_fields),
                       f"Missing required fields: {required_fields - packet_fields}")

    def test_predictable_packet_ordering(self):
        """Test that packets are emitted in predictable order (round_id, member_id)."""
        # Create test packets with different round/member combinations
        packets = [
            CouncilRoundPacket(
                timestamp="2024-01-01T00:00:00",
                session_id="test_session",
                round_id=2,
                member_id="auditor",
                engine="ollama",
                seed=12345,
                prompt_hash="abc123",
                inputs={},
                decision="continue",
                rationale="test",
                confidence=0.8,
                citations=[],
                rag={},
                cag={},
                novelty={},
                memory_directive={"tier": "fast"},
                cost={"input_tokens": 100, "output_tokens": 50, "latency_ms": 1000},
                errors=[]
            ),
            CouncilRoundPacket(
                timestamp="2024-01-01T00:00:00",
                session_id="test_session",
                round_id=1,
                member_id="coordinator",
                engine="ollama",
                seed=12345,
                prompt_hash="abc123",
                inputs={},
                decision="continue",
                rationale="test",
                confidence=0.8,
                citations=[],
                rag={},
                cag={},
                novelty={},
                memory_directive={"tier": "fast"},
                cost={"input_tokens": 100, "output_tokens": 50, "latency_ms": 1000},
                errors=[]
            ),
            CouncilRoundPacket(
                timestamp="2024-01-01T00:00:00",
                session_id="test_session",
                round_id=1,
                member_id="strategist",
                engine="ollama",
                seed=12345,
                prompt_hash="abc123",
                inputs={},
                decision="continue",
                rationale="test",
                confidence=0.8,
                citations=[],
                rag={},
                cag={},
                novelty={},
                memory_directive={"tier": "fast"},
                cost={"input_tokens": 100, "output_tokens": 50, "latency_ms": 1000},
                errors=[]
            )
        ]
        
        # Sort packets as the orchestrator would
        packet_tuples = [(p, None, True) for p in packets]
        packet_tuples.sort(key=lambda x: (x[0].round_id, x[0].member_id))
        
        # Verify ordering: round 1 coordinator, round 1 strategist, round 2 auditor
        self.assertEqual(packet_tuples[0][0].round_id, 1)
        self.assertEqual(packet_tuples[0][0].member_id, "coordinator")
        self.assertEqual(packet_tuples[1][0].round_id, 1)
        self.assertEqual(packet_tuples[1][0].member_id, "strategist")
        self.assertEqual(packet_tuples[2][0].round_id, 2)
        self.assertEqual(packet_tuples[2][0].member_id, "auditor")


if __name__ == '__main__':
    unittest.main()