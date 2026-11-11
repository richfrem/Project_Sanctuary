#!/usr/bin/env python3
"""Post-run checks for MISSION-073 H1(c),(d),H2,H3 against frozen definitions. Reads burnside_n4.jsonl, fix_py.jsonl, summary_n4.json."""
import json
rows = {(tuple(r["perm"]), r["flip"]): r for r in map(json.loads, open("burnside_n4.jsonl"))}
py = [json.loads(l) for l in open("fix_py.jsonl")]
mism = [p for p in py if rows[(tuple(p["perm"]), p["flip"])]["cnt"] != p["cnt"] or rows[(tuple(p["perm"]), p["flip"])]["cycles"] != p["cycles"] or rows[(tuple(p["perm"]), p["flip"])]["det"] != p["det"]]
ident = rows[((0, 1, 2, 3), 0)]
s = json.load(open("summary_n4.json"))
rot, allg = s["rotations"]["orbits_by_k"]["4"], s["all_symmetries"]["orbits_by_k"]["4"]
out = {
 "H1c_elements_compared": len(py), "H1c_mismatches": len(mism),
 "H1d_group_elements": len(rows), "H1d_det_plus": sum(1 for r in rows.values() if r["det"] == 1), "H1d_identity_cycles": ident["cycles"],
 "H1a_integral": s["rotations"]["integral"] and s["all_symmetries"]["integral"], "H1b_axis_symmetry": s["identity_axis_symmetry_ok"],
 "H2_rot_k4": rot, "H2_in_interval": 14400000 <= rot <= 14800000,
 "H3_all_k4": allg, "H3_half_rot": rot / 2, "H3_holds": rot / 2 < allg < rot,
}
out["H1_holds"] = out["H1c_mismatches"] == 0 and out["H1d_group_elements"] == 384 and out["H1d_det_plus"] == 192 and out["H1d_identity_cycles"] == 32 and out["H1a_integral"] and out["H1b_axis_symmetry"]
print(json.dumps(out, indent=1))
