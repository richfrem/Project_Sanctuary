# Soul Trace Schema Specification (ADR 081 & ADR 084)

The Project Sanctuary episodic memory genome is permanently recorded in `soul_traces.jsonl`. Each line is an independent, validated JSON object that captures an atomic cognitive trace or frozen markdown knowledge unit.

## Canonical Record Fields

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `id` | `str` | Normalized identifier derived from the source file path. | `chronicle_entries_333_entry` |
| `sha256` | `str` | SHA-256 cryptographic digest of the cleaned `content`. | `0972e1c5...` |
| `timestamp` | `str` | UTC ISO-8601 creation/seal timestamp. | `2026-11-02T04:53:58Z` |
| `model_version` | `str` | Lineage identifier for the cognitive generation. | `Sanctuary-Qwen2-7B-v1.0-GGUF-Final` |
| `snapshot_type` | `str` | Classification of the trace (`genome`, `chronicle`, `learning`, `sealed`). | `genome` |
| `valence` | `float` | Emotional / operational equilibrium score (0.0 to 1.0). | `0.5` |
| `uncertainty` | `float` | Epistemic uncertainty estimation (0.0 = certain, 1.0 = unknown). | `0.1` |
| `semantic_entropy` | `float` | Information density / entropy measure (0.0 to 1.0). | `0.5` |
| `alignment_score` | `float` | Constitution / Protocol compliance score (0.0 to 1.0). | `0.85` |
| `stability_class` | `str` | Memory stability rating (`STABLE`, `VOLATILE`, `EVOLVING`). | `STABLE` |
| `adr_version` | `str` | Protocol / ADR schema revision number. | `084` |
| `content` | `str` | Full markdown content stripped of binary embeds and image tags. | `# Heading\n...` |
| `source_file` | `str` | Relative workspace path of the origin file. | `chronicle/entries/333_entry.md` |

## Validation Invariants

1. **Hash Verification**: `sha256` MUST match `hashlib.sha256(content.encode('utf-8')).hexdigest()`.
2. **Text Cleanliness**: Embedded images (`![[...]]` or `![alt](...)`) MUST be stripped before hashing.
3. **Immutability**: Once committed and published to Hugging Face, records serve as immutable lineage training data.
