#!/usr/bin/env bash
# council_orchestrator/scripts/dashboard/jq_dashboard.sh
# Phase 2 Council Observability Dashboard
# Usage: ./jq_dashboard.sh /path/to/session_dir [save_snapshot]

SESSION_DIR="${1:-WORK_IN_PROGRESS}"
SAVE_SNAPSHOT="${2:-false}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SNAPSHOT_DIR="scripts/dashboard/snapshots/${TIMESTAMP}"

echo "=== Phase 2 Council Observability Dashboard ==="
echo "Session: $SESSION_DIR"
echo "Timestamp: $(date)"
echo

# Create snapshot directory if saving
if [ "$SAVE_SNAPSHOT" = "true" ]; then
    mkdir -p "$SNAPSHOT_DIR"
    echo "Saving snapshot to: $SNAPSHOT_DIR"
    echo
fi

# Memory Tier Distribution
echo "📊 Memory Tier Distribution:"
TIER_DATA=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | \
jq -r '.memory_directive.tier' | sort | uniq -c | sort -nr)
echo "$TIER_DATA"
if [ "$SAVE_SNAPSHOT" = "true" ]; then
    echo "$TIER_DATA" > "$SNAPSHOT_DIR/tier_distribution.txt"
fi
echo

# Novelty Signal Distribution
echo "🔍 Novelty Signal Distribution:"
NOVELTY_DATA=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | \
jq -r '.novelty.signal' | sort | uniq -c | sort -nr)
echo "$NOVELTY_DATA"
if [ "$SAVE_SNAPSHOT" = "true" ]; then
    echo "$NOVELTY_DATA" > "$SNAPSHOT_DIR/novelty_distribution.txt"
fi
echo

# Conflict Detection
echo "⚠️  Conflict Detection:"
CONFLICT_DATA=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | \
jq -r '.conflict.conflicts_with | length > 0' | grep -c true || echo "0")
TOTAL_PACKETS=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | wc -l | tr -d ' ')
echo "Conflicts detected: $CONFLICT_DATA / $TOTAL_PACKETS packets"
if [ "$TOTAL_PACKETS" -gt 0 ]; then
    CONFLICT_RATE=$(echo "scale=2; $CONFLICT_DATA * 100 / $TOTAL_PACKETS" | bc 2>/dev/null || echo "0")
    echo "Conflict rate: ${CONFLICT_RATE}%"
fi
if [ "$SAVE_SNAPSHOT" = "true" ]; then
    echo "Conflicts: $CONFLICT_DATA / $TOTAL_PACKETS" > "$SNAPSHOT_DIR/conflict_stats.txt"
fi
echo

# Performance Metrics
echo "⚡ Performance Analysis:"
LATENCY_DATA=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | \
jq -r '.retrieval.retrieval_latency_ms' | awk 'BEGIN {sum=0; count=0; max=0} {sum+=$1; count++; if($1>max) max=$1} END {if(count>0) printf "Mean: %.1fms\nP95: ?\nMax: %dms\nCount: %d\n", sum/count, max, count}')
echo "$LATENCY_DATA"

STAGE_DATA=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | \
jq -r '[.retrieval.plan_latency_ms, .retrieval.analyze_latency_ms, .retrieval.emit_latency_ms] | @csv' | \
awk -F, 'BEGIN {p=0; a=0; e=0; c=0} {p+=$1; a+=$2; e+=$3; c++} END {if(c>0) printf "Stage Latencies (avg): Plan=%.1fms, Analyze=%.1fms, Emit=%.1fms\n", p/c, a/c, e/c}')
echo "$STAGE_DATA"
if [ "$SAVE_SNAPSHOT" = "true" ]; then
    echo "$LATENCY_DATA" > "$SNAPSHOT_DIR/performance_metrics.txt"
    echo "$STAGE_DATA" >> "$SNAPSHOT_DIR/performance_metrics.txt"
fi
echo

# Evidence Quality
echo "📚 Evidence Quality:"
CITATIONS_DATA=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | \
jq -r 'select(.citations | length > 0) | .memory_directive.tier' | grep -c -E "(medium|slow)" || echo "0")
TOTAL_CITATIONS=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | \
jq -r '.citations | length > 0' | grep -c true || echo "0")
echo "Packets with citations: $TOTAL_CITATIONS"
echo "Citations promoted beyond fast: $CITATIONS_DATA"
if [ "$TOTAL_CITATIONS" -gt 0 ]; then
    PROMOTION_RATE=$(echo "scale=1; $CITATIONS_DATA * 100 / $TOTAL_CITATIONS" | bc 2>/dev/null || echo "0")
    echo "Evidence promotion rate: ${PROMOTION_RATE}%"
fi
if [ "$SAVE_SNAPSHOT" = "true" ]; then
    echo "Citations: $TOTAL_CITATIONS, Promoted: $CITATIONS_DATA" > "$SNAPSHOT_DIR/evidence_quality.txt"
fi
echo

# Cache Performance
echo "💾 Cache Performance:"
CACHE_DATA=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | \
jq -r '.cag | select(.) | "\(.hit_streak // 0) \(.ema_7d // 0)"' | \
awk 'BEGIN {hits=0; total_ema=0; count=0} {hits+=$1; total_ema+=$2; count++} END {if(count>0) printf "Avg Hit Streak: %.1f\nAvg EMA 7d: %.3f\nCache Queries: %d\n", hits/count, total_ema/count, count}')
echo "$CACHE_DATA"
if [ "$SAVE_SNAPSHOT" = "true" ]; then
    echo "$CACHE_DATA" > "$SNAPSHOT_DIR/cache_performance.txt"
fi
echo

# Phase 3 Readiness
echo "🚀 Phase 3 Promotion Candidates:"
PROMOTION_DATA=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | \
jq -r 'select(.memory_directive.tier == "slow" and (.cag.hit_streak // 0) >= 3) | "\(.member_id): \(.decision) (streak: \(.cag.hit_streak // 0), ema: \(.cag.ema_7d // 0))"' | \
head -5)
if [ -n "$PROMOTION_DATA" ]; then
    echo "$PROMOTION_DATA"
else
    echo "No candidates ready for Phase 3 promotion yet"
fi
if [ "$SAVE_SNAPSHOT" = "true" ]; then
    echo "$PROMOTION_DATA" > "$SNAPSHOT_DIR/phase3_candidates.txt"
fi
echo

# SLO Compliance
echo "📏 SLO Compliance Check:"
if [ "$TOTAL_PACKETS" -gt 0 ]; then
    # Calculate basic SLO metrics
    HIGH_LATENCY=$(find "$SESSION_DIR" -name "round_*.jsonl" -exec cat {} \; 2>/dev/null | \
    jq -r '.retrieval.retrieval_latency_ms' | awk '$1 > 150 {count++} END {print count+0}')
    SLO_VIOLATIONS=$HIGH_LATENCY
    echo "Packets violating p95 latency SLO (>150ms): $SLO_VIOLATIONS"
    echo "SLO compliance: $(echo "scale=1; ($TOTAL_PACKETS - $SLO_VIOLATIONS) * 100 / $TOTAL_PACKETS" | bc 2>/dev/null || echo "100")%"
fi

echo
echo "=== Dashboard Complete ==="
if [ "$SAVE_SNAPSHOT" = "true" ]; then
    echo "Snapshot saved to: $SNAPSHOT_DIR"
fi