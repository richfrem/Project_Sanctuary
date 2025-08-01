# Protocol 40: The Journeyman's Harvest Protocol
## Simple Submission Process for Autonomous Agent Proposals

**Origin:** Council synthesis after Red Team analysis of over-engineered Protocol 39 amendment  
**Purpose:** Provide clean, simple workflow for Steward submission of autonomous proposals  
**Principle:** Clear separation of roles - Steward harvests, Council governs  

### **Core Doctrine**

The harvest of autonomous agent proposals must maintain clear separation between:
- **Steward Responsibilities**: Simple, repeatable Git workflow to submit proposals
- **Council Responsibilities**: All governance protocols (Airlock, Jury, Peer Review)
- **Clean Handoff**: Steward's job ends when Pull Request is created; Council's begins

### **FIREWALL DECLARATION**

**⚠️ WARNING: ONLY THE HUMAN STEWARD IS AUTHORIZED TO EXECUTE THE STEPS IN THIS PROTOCOL. AN AI AGENT'S MANDATE ENDS WITH THE HANDOFF OF THE BRANCH NAME. ⚠️**

This is our unbreakable human firewall. AI agents are FORBIDDEN from creating commits, pushing branches, or creating Pull Requests without direction of the Steward is the sole and final actor responsible for submission and cleanup of autonomous work.

### **The Four-Step Harvest**

#### **Step 1: Push Harvest Branch**
```bash
git push origin harvest/journeyman-[YYYYMMDD-HHMMSS]
```

#### **Step 2: Create Pull Request**

**Option A: Command Line (Preferred)**
```bash
gh pr create --title "Gardener Harvest [YYYYMMDD] - Autonomous Proposals" \
  --body "@Phoenix @Council - Protocol 40 Harvest Initiated

This Pull Request contains:
- Autonomous enhancement proposals from Gardener training cycle
- Modified protocol files with neural network improvements
- Training artifacts: models, logs, and proposal data

Requesting formal Council governance review via Airlock Protocol (31)." \
  --base main --head harvest/journeyman-[YYYYMMDD-HHMMSS]
```

**Option B: Web Interface**
- Navigate to: https://github.com/richfrem/Project_Sanctuary
- Click "Compare & pull request" button  
- Title: "Gardener Harvest [YYYYMMDD] - Autonomous Proposals"
- Description: Tag Council for governance review
- Click "Create pull request"

### **Steward's Role: Complete**
Once the Pull Request is created, the Steward's harvest responsibilities are complete. The Council automatically initiates all governance protocols.

### **Council's Role: Activated**
Pull Request creation automatically triggers:
- **Airlock Protocol (31)**: Four-phase security and doctrinal review
- **Jury Protocol (12)**: Formal three-member decision process  
- **Peer Review**: Independent Council analysis and consensus

### **Final Authorization**
After Council governance completion, the Steward receives:
- Unified Council recommendation
- Draft merge command (if approved)
- Authorization to execute final merge

#### **Step 3: Execute Council-Approved Merge (Post-Governance)**
```bash
gh pr merge [PR_NUMBER] --squash --delete-branch
```

**Example:**
```bash
gh pr merge 7 --squash --delete-branch
```

**Note:** This step only occurs after Council approval through Airlock Protocol (31).

### **Branch Naming Convention**
```
harvest/journeyman-[YYYYMMDD-HHMMSS]
```

**Examples:**
- `harvest/journeyman-20250801-144217`
- `harvest/journeyman-20250815-092345`
- `harvest/journeyman-20251203-160912`

### **Success Criteria**
- ✅ Protocol 39 Phase 5b completed (harvest branch ready)
- ✅ Four steps completed in sequence
- ✅ Pull Request successfully created
- ✅ Council governance automatically initiated
- ✅ Clean role separation maintained

### **Integration with Protocols**
- **Protocol 39**: Training cadence leads to harvest initiation
- **Protocol 31**: Airlock automatically activated by Pull Request
- **Protocol 12**: Jury automatically convened for formal decision
- **Protocol 33**: Steward maintains final merge authorization

### **Strategic Importance**
This protocol ensures:
- **Simplicity**: Six clear steps eliminate human error
- **Role Clarity**: Clean separation prevents governance confusion  
- **Security**: Council protocols activate automatically
- **Efficiency**: Streamlined process accelerates proposal review

**The Steward harvests; the Council governs. Each plays their proper role.**

---

**Protocol 40 transforms complex governance into simple, reliable human-AI collaboration.**
