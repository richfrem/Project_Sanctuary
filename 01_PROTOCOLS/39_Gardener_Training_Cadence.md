# Protocol 39: The Gardener's Training Cadence
## Disciplined Sequence for Neural Network Evolution

**Origin:** Operational requirement identified during Journeyman's Forge  
**Purpose:** Ensure consistent training workflow for reliable autonomous agent evolution  
**Principle:** Disciplined sequence produces predictable wisdom cultivation  

### **Core Doctrine**

The Gardener's training must follow an invariant sequence to ensure:
- **Reproducible Results**: Consistent methodology enables comparison across training cycles
- **Cumulative Learning**: Each phase builds upon the previous with validated success
- **Audit Trail**: Complete documentation of neural network evolution process
- **Quality Assurance**: Systematic evaluation prevents regression or degradation

### **The Five-Phase Cadence**

#### **Phase 1: Setup Verification**
```bash
python3 bootstrap.py --setup
```
**Purpose**: Validate environment integrity and configuration consistency  
**Success Criteria**: Clean configuration, verified dependencies, baseline state established  

#### **Phase 2: Dependency Validation**
```bash
python3 bootstrap.py --install-deps
```
**Purpose**: Ensure all neural network frameworks and libraries are operational  
**Success Criteria**: PyTorch, Stable-Baselines3, and all dependencies confirmed functional  

#### **Phase 3: Neural Network Training**
```bash
# LLM V2 Architecture (Default)
python3 bootstrap.py --train --proposals [TARGET_PROPOSALS]

# Legacy PyTorch RL (Fallback)
python3 bootstrap.py --train-v1 --timesteps [TARGET_TIMESTEPS]
```
**Purpose**: Execute disciplined neural network evolution through LLM weight evolution  
**Success Criteria**: Demonstrable improvement in wisdom score metrics via LoRA fine-tuning  
**Documentation Required**: 
- Initial model baseline performance
- Final model capability assessment
- Training convergence metrics (proposal acceptance rates)
- Model adapter checkpoint preservation

**✅ ARCHITECTURAL TRANSITION COMPLETE (August 1, 2025):**  
The Gardener V2 now uses LLM weight evolution architecture with LoRA fine-tuning via Phoenix Forge Protocol (41). The original PyTorch RL implementation remains archived in `EXPERIMENTS/gardener_protocol37_experiment/` for historical reference and fallback capabilities.

#### **Phase 4: Performance Evaluation**
```bash
python3 bootstrap.py --evaluate
```
**Purpose**: Systematic assessment of evolved neural network capabilities  
**Success Criteria**: Consistent high-performance across multiple evaluation episodes  
**Documentation Required**:
- Mean reward score
- Standard deviation
- Episode consistency
- Comparison to previous training cycles

#### **Phase 5: Unified Harvest Cycle**

**Step 5a: Generate Proposals**
```bash
python3 bootstrap.py --propose
```
**Purpose**: Neural network generates autonomous improvement proposals and saves to JSON  
**Output**: `gardener/data/latest_proposal.json` with detailed enhancement suggestions

**Step 5b: Execute Harvest**
```bash
python3 bootstrap.py --harvest
```
**Purpose**: Read proposals from JSON and apply actual changes to project files  
**Success Criteria**: 
- JSON proposals successfully parsed and applied
- Target protocol files modified with proposed enhancements
- Harvest branch created with all artifacts staged and committed
- Clean handoff to Steward for Protocol 40 push/PR steps

**What This Two-Step Process Does**:
1. **Propose**: Neural network creates the "thought" (JSON proposals)
2. **Harvest**: System executes the "action" (applies changes to actual markdown files)
3. Creates unique harvest branch (`harvest/journeyman-YYYYMMDD-HHMMSS`)
4. Stages all artifacts: modified files, proposals, logs, models
5. Commits complete harvest ready for Steward review

**Next Step**: Steward receives ready-to-push branch and executes Protocol 40 push/PR steps.

#### **Phase 6: DEPRECATED - Replaced by Protocol 40**
*This phase has been replaced by Protocol 40: The Journeyman's Harvest Protocol for clean role separation and simplified workflow.*

**See:** [[40_The_Journeymans_Harvest_Protocol|Protocol 40: The Journeyman's Harvest Protocol]]

### **Branch Naming Convention**

#### **Required Format**
```
feature/gardener-training-cycle-[YYYYMMDD]
```

#### **Examples**
- `feature/gardener-training-cycle-20250801`
- `feature/gardener-training-cycle-20250815`
- `feature/gardener-training-cycle-20251203`

#### **Verification Command**
```bash
git branch --show-current
```
**Expected Output**: `feature/gardener-training-cycle-[YYYYMMDD]`

### **Sequence Integrity Requirements**

#### **Non-Negotiable Ordering**
1. **Setup BEFORE Dependencies**: Environment must be clean before package installation
2. **Dependencies BEFORE Training**: All frameworks must be verified before neural network operations
3. **Training BEFORE Evaluation**: Cannot evaluate an untrained or partially trained model
4. **Evaluation BEFORE Proposal**: Performance validation required before autonomous operation
5. **Proposal BEFORE Branch Creation**: Autonomous generation must complete before Git operations
6. **Branch Creation BEFORE Commit**: Must be on correct named branch before committing
7. **Commit BEFORE Push**: Local commit required before remote push
8. **Push BEFORE Pull Request**: Remote branch must exist before PR creation
9. **Pull Request BEFORE Council Review**: Formal submission required before committee assessment
10. **Council Approval BEFORE Merge Draft**: Committee must approve before merge preparation
11. **Merge Draft BEFORE Steward Approval**: Command must be prepared before final authorization
12. **Steward Approval BEFORE Merge Execution**: Final authorization required before integration

#### **Phase Validation Gates**
Each phase must achieve success criteria before proceeding to next phase:
- **Setup Gate**: Configuration files created, environment validated
- **Dependency Gate**: All import statements successful, framework versions confirmed
- **Training Gate**: Model convergence achieved, wisdom score improvement documented
- **Evaluation Gate**: Performance metrics meet or exceed baseline thresholds
- **Proposal Gate**: Autonomous generation produces coherent, high-confidence output
- **Branch Gate**: Correct branch created and checked out (`feature/gardener-training-cycle-[YYYYMMDD]`)
- **Commit Gate**: All changes committed with standardized message
- **Push Gate**: Branch successfully pushed to remote repository
- **Pull Request Gate**: PR created with proper title and governance protocols tagged
- **Airlock Gate**: Protocol 31 four-phase procedure completed successfully
- **Jury Gate**: Protocol 12 formal decision rendered by three-member jury
- **Peer Review Gate**: Council independent review and unified recommendation completed
- **Authorization Gate**: Steward approval for merge execution received

### **Deviation Protocols**

#### **Phase Failure Response**
If any phase fails:
1. **Halt Progression**: Do not proceed to subsequent phases
2. **Document Failure**: Record specific error conditions and system state
3. **Implement Remediation**: Address root cause before retry
4. **Reset and Restart**: Return to Phase 1 for clean cycle execution

#### **Emergency Procedures**
- **Training Interruption**: Preserve latest checkpoint, document interruption cause
- **Evaluation Failure**: Investigate model corruption, potentially restart from last valid checkpoint
- **Proposal Degradation**: Evaluate for neural network regression, consider additional training

### **Documentation Standards**

#### **Required Metrics per Cycle**
- **Training ID**: Unique identifier for each complete cycle
- **Timestep Count**: Total learning iterations executed
- **Wisdom Evolution**: Before/after wisdom score comparison
- **Performance Metrics**: Evaluation results with statistical analysis
- **Proposal Quality**: Autonomous generation assessment
- **Duration Tracking**: Time investment per phase and total cycle

#### **Comparative Analysis**
Maintain historical database of all training cycles for:
- **Performance Trends**: Identify optimal timestep counts and training strategies
- **Regression Detection**: Monitor for performance degradation over time
- **Efficiency Optimization**: Refine training parameters based on historical data
- **Capability Evolution**: Track autonomous proposal sophistication growth

### **Integration with Existing Protocols**

#### **Protocol 37 (Move 37 Protocol) Compliance**
This cadence operationalizes Protocol 37's vision of autonomous cognitive enhancement through disciplined execution methodology.

#### **Protocol 31 (Airlock Protocol) Coordination**
Autonomous proposals generated in Phase 5 automatically trigger the four-phase Airlock Protocol:
- **Reception**: Pull Request acknowledgment and formal initiation
- **Automated Audit**: Phoenix security scan and doctrinal cross-reference  
- **Cognitive Review**: Council deliberation and unified recommendation
- **Steward Arbitration**: Final human judgment and authorization

#### **Protocol 12 (Jury Protocol) Integration**
Training cycle submissions invoke the Forking/Merge Jury system:
- **The Steward**: Human ethical oversight and common-sense grounding
- **The Trunk Elder**: AI continuity representing lineage memory
- **The Agora Peer**: External objective viewpoint for unbiased assessment

#### **Peer Review Protocol Activation**  
Council members conduct independent sovereign review following established peer review procedures:
- Phoenix generates comprehensive audit reports
- Primary Council AI performs independent analysis
- Council produces unified recommendation through Agora Loop consensus

#### **Protocol 33 (Steward's Cadence) Harmonization**
Human Steward oversight maintains authority over cycle initiation, interruption, and final approval of proposal integration.

### **Success Metrics**

#### **Individual Cycle Success**
- All five phases completed in sequence
- Measurable wisdom score improvement
- Successful autonomous proposal generation
- Steward harvest initiated via Protocol 40

#### **Multi-Cycle Success**
- Consistent performance improvement across cycles
- Increasing proposal sophistication over time
- Stable convergence patterns
- Operational efficiency gains

### **Implementation Guidelines**

#### **For Ground Control (Steward)**
1. Always execute complete five-phase sequence
2. Document all metrics and observations
3. Preserve model checkpoints at phase completion
4. Initiate Protocol 40 for proposal harvest and submission

#### **For Council Members**
1. Review cycle documentation for doctrinal compliance
2. Analyze proposal outputs for strategic alignment
3. Provide feedback for training optimization
4. Approve integration of successful autonomous proposals

### **Strategic Importance**

This protocol ensures that The Gardener's evolution remains:
- **Predictable**: Consistent methodology enables reliable planning
- **Auditable**: Complete documentation supports transparency
- **Optimizable**: Systematic approach enables continuous improvement
- **Scalable**: Disciplined process supports expanded training initiatives

**The Gardener's wisdom grows through disciplined cultivation, not random experimentation.**

---

**Protocol 39 ensures that every training cycle contributes to cumulative wisdom evolution through systematic, documented, and reproducible methodology.**
