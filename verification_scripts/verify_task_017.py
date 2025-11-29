
import sys
import os
from pathlib import Path
from mcp_servers.orchestrator.server import orchestrator_run_strategic_cycle

def verify_task_017():
    print("--- Starting Task #017 Verification ---")
    
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # 1. Create Dummy Research Report
    report_path = project_root / "WORK_IN_PROGRESS" / "strategic_gap_report.md"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text("# Strategic Gap: Test\n\nWe need to test the loop.")
    
    print(f"\n1. Created Dummy Report: {report_path}")
    
    # 2. Run Strategic Cycle
    print("\n2. Running Strategic Cycle...")
    try:
        result = orchestrator_run_strategic_cycle(
            gap_description="Testing the autonomous loop",
            research_report_path=str(report_path),
            days_to_synthesize=1
        )
        print("\n--- Result Output ---")
        print(result)
        
        if "[CRITICAL FAIL]" in result:
            print("\n[FAIL] Cycle failed.")
        else:
            print("\n[SUCCESS] Cycle completed successfully.")
            
    except Exception as e:
        print(f"\n[FAIL] Execution error: {e}")
        import traceback
        traceback.print_exc()
        
    # Cleanup
    if report_path.exists():
        report_path.unlink()

if __name__ == "__main__":
    verify_task_017()
