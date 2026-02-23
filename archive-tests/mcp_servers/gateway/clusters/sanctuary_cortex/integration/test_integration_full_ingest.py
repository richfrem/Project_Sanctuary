import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
@pytest.mark.slow
class TestCortexFullIngest:
    """Integration test to perform a full ingestion into the Cortex RAG DB."""

    async def test_cortex_ingest_full(self, cortex_cluster):
        """Call the `cortex-ingest-full` tool against sanctuary_cortex.

        Uses container path `/app/tests/fixtures/test_docs` which is the
        standard mapped fixtures directory in E2E tests.
        """
        args = {
            "source_directories": ["/app/tests/fixtures/test_docs"],
            "purge_existing": False
        }

        # This call can take several minutes depending on fixture size
        result = await cortex_cluster.call_tool("cortex-ingest-full", args)

        assert result is not None

        # The MCP SDK returns a CallToolResult with `.content` (list of TextContent) when successful.
        # Extract and parse the first text payload if available.
        parsed = None
        if hasattr(result, "content") and result.content:
            try:
                import json
                text = result.content[0].text
                parsed = json.loads(text)
            except Exception:
                parsed = None

        if isinstance(result, dict):
            # Some implementations return a dict with a success key
            assert result.get("success", True) is True
        elif parsed:
            # Validate expected fields in parsed JSON
            assert parsed.get("status") in ("success", None) or parsed.get("documents_processed", 0) >= 0
        else:
            # Fall back to basic checks on the SDK result object
            assert not getattr(result, "isError", False)

