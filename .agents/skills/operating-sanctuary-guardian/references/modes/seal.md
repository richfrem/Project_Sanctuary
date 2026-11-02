# Mode: Seal (`seal`)

Executes the Protocol 128 Technical Seal and Session Retrospective.

## Execution Steps

1. Run the technical learning debrief to capture recent changes:
```bash
python3 scripts/learning_debrief.py --hours 24
```
2. Execute the Technical Seal snapshot:
```bash
python3 scripts/capture_snapshot.py --type seal --context "Session closure and milestone completion"
```
3. Verify that the sealed snapshot bundle is generated under `context/` or `.agent/learning/`.
