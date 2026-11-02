# Protocol 128: The Learning Loop & Guardian Lifecycle

Protocol 128 defines Project Sanctuary's continuous lifecycle for cognitive continuity across agent sessions.

## The 8 Phases of Protocol 128

1. **Phase I: Bootloader & Orientation**:
   - Run `guardian_wakeup.py` to inspect tactical priorities and active tasks.
   - Load cognitive primer and environment contracts from `.agent/learning/`.
   - Perform the Iron Check to verify zero state drift.

2. **Phase II: Strategic Formulation**:
   - Define task bounds and specifications.
   - For ambiguous features, outline clear invariants and requirements.

3. **Phase III: Cognitive Execution**:
   - Execute implementation using approved coding and testing practices.
   - Maintain dual-layer documentation and type safety.

4. **Phase IV: Inner Loop & Review**:
   - Validate implementation against unit and regression tests.
   - Perform pre-completion verification checks.

5. **Phase V: Retrospective**:
   - Capture lessons learned, friction events, and architectural insights.
   - Record retrospective reflections.

6. **Phase VI: Technical Seal**:
   - Run `capture_snapshot.py --type seal` to generate the immutable session snapshot.
   - Verify repository state and ensure no uncommitted artifacts remain unstamped.

7. **Phase VII: Soul Persistence**:
   - Export structured vault notes into JSONL schema matching Hugging Face dataset format.
   - Persist learning traces to the external dataset via `persist_soul.py` or global `managing-huggingface-assets`.

8. **Phase VIII: Final Seal & Closure**:
   - Clean ephemeral session logs.
   - Return clean environment ready for the next session.
