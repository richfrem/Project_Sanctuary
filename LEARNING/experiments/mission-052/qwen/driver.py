import json,re,subprocess,sys,time,os
R="/Users/richardfremmerlid/Projects/Project_Sanctuary"; W=R+"/.worktrees/qwen-local-mission-052"
OUT=W+"/LEARNING/experiments/mission-052/qwen"; os.makedirs(OUT,exist_ok=True)
ids=["004","007","011","023","036","039","040","041","042","045","046","050"]
res=[]
for i in ids:
    d=json.load(open(f"{W}/LEARNING/missions/MISSION-{i}.json"))
    truth=sorted(set(re.findall(r"\d+(?:\.\d+)?",d["falsification_criteria"])))
    task=(f"Read the file LEARNING/missions/MISSION-{i}.json. In its falsification_criteria field, list every number that appears in the text. "
          'Reply in <final_answer> as JSON exactly like {"numbers": ["50","5"]} with numbers as strings in order of appearance.')
    t=time.time()
    p=subprocess.run([sys.executable,R+"/plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/forge_llm.py","loop",task,"--root",W,"--max-turns","3","--json"],capture_output=True,text=True,timeout=300)
    raw=p.stdout
    got=None;err=None
    try:
        m=re.findall(r'\{\s*\\?"numbers\\?"\s*:\s*\[[^\]]*\]\s*\}',raw)
        obj=json.loads(m[-1].replace('\\"','"')) if m else None
        got=sorted(set(map(str,obj["numbers"]))) if obj else None
    except Exception as e: err=repr(e)
    res.append({"mission":i,"truth":truth,"got":got,"pass":got==truth,"seconds":round(time.time()-t,1),"rc":p.returncode,"err":err,"raw_tail":raw[-600:]})
    json.dump(res,open(OUT+"/results.json","w"),indent=1)
print("pass",sum(r["pass"] for r in res),"of",len(res))
