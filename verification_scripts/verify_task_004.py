
import sys
import json
import os
from pathlib import Path
from mnemonic_cortex.app.synthesis.generator import SynthesisGenerator
from mnemonic_cortex.app.training.versioning import VersionManager

def verify_task_004():
    print("--- Starting Task #004 Verification ---")
    
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # 1. Test Synthesis Generator
    print("\n1. Testing Synthesis Generator...")
    generator = SynthesisGenerator(str(project_root))
    
    # Create a dummy protocol file to ensure we have data
    dummy_proto = project_root / "01_PROTOCOLS" / "999_Test_Protocol.md"
    dummy_proto.parent.mkdir(exist_ok=True)
    dummy_proto.write_text("# Protocol 999: Test\n\nThis is a test protocol for synthesis.")
    
    try:
        packet = generator.generate_packet(days=1)
        print(f"   [SUCCESS] Packet generated with ID: {packet.packet_id}")
        print(f"   [INFO] Found {len(packet.source_ids)} source documents.")
        
        output_path = generator.save_packet(packet)
        print(f"   [SUCCESS] Packet saved to: {output_path}")
        
        # Verify content
        with open(output_path, "r") as f:
            data = json.load(f)
            if "999_Test_Protocol.md" in str(data["source_ids"]):
                print("   [SUCCESS] Dummy protocol found in packet.")
            else:
                print("   [WARN] Dummy protocol NOT found in packet source_ids.")
                
    except Exception as e:
        print(f"   [FAIL] Generator failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Test Versioning
    print("\n2. Testing Version Manager...")
    manager = VersionManager(str(project_root))
    version = manager.register_adapter(
        packet_id=packet.packet_id,
        base_model="test-model",
        path=str(project_root / "mnemonic_cortex/adaptors/test_adapter.npz")
    )
    print(f"   [SUCCESS] Registered version: {version}")
    
    next_ver = manager.get_next_version()
    print(f"   [INFO] Next version would be: {next_ver}")
    
    # Cleanup
    if dummy_proto.exists():
        dummy_proto.unlink()

if __name__ == "__main__":
    verify_task_004()
