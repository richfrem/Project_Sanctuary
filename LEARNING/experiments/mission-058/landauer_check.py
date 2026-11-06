"""MISSION-058 reproducible computation: the Landauer floor versus real storage energy, and what 'deleting a record' physically changes.
Part A: arithmetic. Part B: an SQLite experiment (logical DELETE vs secure_delete). Run: python3 landauer_check.py"""
import json, math, os, sqlite3, tempfile
from pathlib import Path

K_B = 1.380649e-23                      # J/K (exact, SI 2019)
OUT = {}

# ---- Part A: arithmetic
OUT["landauer_J_per_bit"] = {f"{T}K": K_B * T * math.log(2) for T in (250, 300, 350)}
L300 = OUT["landauer_J_per_bit"]["300K"]
bits_tb = 8e12                                            # 1 TB (10^12 bytes)
OUT["landauer_cost_to_erase_1TB_J_at_300K"] = bits_tb * L300
real = {                                                   # per-bit energies taken from the cited sources (see findings.md); NOT measured here
    "embedded flash program, 122 pJ/bit (Jeloka et al., VLSI 2015, SONOS)": 122e-12,
    "whole-SSD average ~1 nJ/bit (Stanford ph240 course page: ~7 W at ~0.7 GB/s -> %.2e J/bit)" % (7 / (0.7e9 * 8)): 7 / (0.7e9 * 8),
    "silicon transistor switching ~1e-15 J (secondary source)": 1e-15,
}
OUT["ratio_to_landauer_300K"] = {k: v / L300 for k, v in real.items()}
OUT["post_scenario"] = {"rack_600W_for_1_hour_J": 600 * 3600, "landauer_floor_to_erase_1TB_J": bits_tb * L300,
                        "rack_hour_over_landauer_1TB": (600 * 3600) / (bits_tb * L300)}
OUT["row_delete_example"] = {"row_bytes": 1024, "landauer_floor_J_if_every_payload_bit_were_erased": 1024 * 8 * L300}

# ---- Part B: does a logical DELETE erase the payload?
def marker_count(path, marker):
    return Path(path).read_bytes().count(marker)

def run(secure):
    d = tempfile.mkdtemp(); p = os.path.join(d, "t.db"); marker = b"MARKER-PAYLOAD-0123456789"
    con = sqlite3.connect(p); con.execute("PRAGMA journal_mode=DELETE"); con.execute(f"PRAGMA secure_delete={'ON' if secure else 'OFF'}")
    con.execute("CREATE TABLE r(id INTEGER PRIMARY KEY, body BLOB)")
    con.executemany("INSERT INTO r(body) VALUES (?)", [(marker + b"." * 200 + str(i).encode(),) for i in range(2000)]); con.commit()
    before = Path(p).read_bytes(); n_before = before.count(marker)
    con.execute("DELETE FROM r"); con.commit(); con.close()
    after = Path(p).read_bytes(); n_after = after.count(marker)
    same = sum(a == b for a, b in zip(before, after)); total = min(len(before), len(after))
    return {"secure_delete": secure, "rows": 2000, "marker_copies_before": n_before, "marker_copies_after_DELETE": n_after,
            "file_bytes": len(after), "fraction_of_bytes_unchanged_by_DELETE": same / total}
OUT["sqlite_logical_delete"] = [run(False), run(True)]
Path("landauer-results.json").write_text(json.dumps(OUT, indent=1))
print(json.dumps({k: OUT[k] for k in ("landauer_J_per_bit", "landauer_cost_to_erase_1TB_J_at_300K", "ratio_to_landauer_300K", "post_scenario")}, indent=1, default=str))
print(json.dumps(OUT["sqlite_logical_delete"], indent=1))
