# Mode: Audit (`audit`)

Verifies environment health and performs the Iron Check against configuration drift.

## Execution Steps

1. Run the audit mode:
```bash
python3 scripts/capture_snapshot.py --type audit
```
2. Verify system traffic light status:
```bash
python3 scripts/guardian_wakeup.py --mode HOLISTIC
```
3. If drift or corruption is detected, enter Safe Mode and alert the operator.
