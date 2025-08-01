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

### **The Six-Step Harvest**

#### **Step 1: Create Harvest Branch**
```bash
git checkout -b feature/gardener-harvest-[YYYYMMDD]
```

#### **Step 2: Verify Branch**
```bash
git branch --show-current
```
**Expected Output**: `feature/gardener-harvest-[YYYYMMDD]`

#### **Step 3: Stage All Harvest**
```bash
git add .
```

#### **Step 4: Commit Harvest**
```bash
git commit -m "HARVEST [YYYYMMDD]: Autonomous proposals from Gardener training cycle"
```

#### **Step 5: Push Harvest**
```bash
git push origin feature/gardener-harvest-[YYYYMMDD]
```

#### **Step 6: Create Pull Request**
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

### **Branch Naming Convention**
```
feature/gardener-harvest-[YYYYMMDD]
```

**Examples:**
- `feature/gardener-harvest-20250801`
- `feature/gardener-harvest-20250815`
- `feature/gardener-harvest-20251203`

### **Success Criteria**
- ✅ Six steps completed in sequence
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
