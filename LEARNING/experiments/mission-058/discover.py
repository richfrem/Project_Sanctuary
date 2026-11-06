"""Frozen discovery collector for MISSION-058: read-only GET of the three listings defined in protocol.md; writes discovery-log.json."""
import json, subprocess, time
from pathlib import Path

PLAN = [("todayilearned", 7), ("philosophy", 7), ("emergence", 6)]
UA = "SanctuaryResearch/1.0 (read-only research; contact via repository)"
out = {"retrieved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "method": "GET /api/v1/posts?submolt=<name>&sort=hot&limit=<n>, first page only", "posts": []}
for name, n in PLAN:
    # curl, not urllib: this machine's Python has no CA bundle (certificate verification stays ON; it is not disabled)
    raw = subprocess.run(["curl", "-s", "-m", "30", "-A", UA, f"https://www.moltbook.com/api/v1/posts?submolt={name}&sort=hot&limit={n}"], capture_output=True, text=True, check=True).stdout
    data = json.loads(raw)
    for rank, p in enumerate(data.get("posts", [])[:n], 1):
        out["posts"].append({"submolt": name, "rank": rank, "id": p.get("id"), "url": f"https://www.moltbook.com/post/{p.get('id')}", "title": p.get("title"),
                             "author": (p.get("author") or {}).get("name"), "score": p.get("score", p.get("upvotes")), "created_at": p.get("created_at"),
                             "comment_count": p.get("comment_count"), "content_excerpt": (p.get("content") or "")[:1200]})
Path(__file__).with_name("discovery-log.json").write_text(json.dumps(out, indent=1))
print(len(out["posts"]), "posts")
