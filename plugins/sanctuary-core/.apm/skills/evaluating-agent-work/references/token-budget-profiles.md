# Token Budget Profiles & Quota Management

This reference documents real-world subscription tiers and rate-limiting guidelines.

---

## 1. The Human Partner's Real-World Constraints
AI services are governed by monthly subscription plans with finite rate limits:
- **ChatGPT Plus**: ~$20/month. Strict rolling message caps per 3-hour window.
- **Claude Pro**: ~$20/month. Message caps per 5-hour window, heavily throttled by context size when using Sonnet.
- **Gemini Pro (AGY)**: ~$20/month. Generous tokens but subject to Requests Per Day (RPD) and Requests Per Minute (RPM).

---

## 2. Best Practices for Token Conservation
1. **Single-Pass Implementations**: Avoid piecemeal micro-iterations that resend the entire codebase repeatedly.
2. **Context Progressive Disclosure**: Retrieve Tier 1 and Tier 2 summaries instead of dumping multi-megabyte files into active context.
3. **Tripwire Bounds**: Never allow an agent or subagent to loop more than 3 times without observable code mutation or new information gain.
4. **Clean Checkpointing**: When quota limits are approached, pause cleanly rather than rushing a half-baked commit.

---

## 3. Measured data supersedes these estimates
The tiers above are static estimates. Real windows and per-task usage are measured at every mission checkout/checkin (see [usage-telemetry-sources.md](usage-telemetry-sources.md)); check the live gauge with `scripts/usage_probe.py headroom` before dispatching.
