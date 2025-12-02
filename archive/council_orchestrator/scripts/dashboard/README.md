# Council Orchestrator Observability Dashboard

Phase 2 observability tools for monitoring council performance, safety, and Phase 3 readiness.

## Usage

```bash
# Basic dashboard
./scripts/dashboard/jq_dashboard.sh /path/to/session/dir

# Save snapshot for trend analysis
./scripts/dashboard/jq_dashboard.sh /path/to/session/dir true
```

## Metrics Tracked

### Memory Tier Distribution
- Fast/Medium/Slow tier assignments
- Evidence quality impact on promotion

### Novelty Analysis
- Novelty signal distribution (none/low/medium/high)
- Raw overlap metrics (token/Jaccard/ROUGE)

### Conflict Detection
- Conflict rate across sessions
- Human-readable conflict reasons

### Performance Analysis
- Per-stage latencies (plan/retrieve/analyze/emit)
- SLO compliance (p95 latency targets)
- Cache performance metrics

### Evidence Quality
- Citation integrity validation
- Evidence promotion rates
- PII redaction effectiveness

### Phase 3 Readiness
- Cache EMA trends
- Promotion candidate identification
- Hit streak analysis

## Snapshot Saving

When `save_snapshot=true`, metrics are saved to:
```
scripts/dashboard/snapshots/YYYYMMDD_HHMMSS/
├── tier_distribution.txt
├── novelty_distribution.txt
├── conflict_stats.txt
├── performance_metrics.txt
├── evidence_quality.txt
├── cache_performance.txt
└── phase3_candidates.txt
```

Use snapshots to track trends over time and validate Phase 3 promotion logic.