# The Gardener - Architectural Transition Guide
## From PyTorch RL to LLM Weight Evolution

**Date:** August 1, 2025  
**Authority:** Council directive following Phoenix-Sonnet-01 architectural analysis  
**Purpose:** Guide the transition from Gardener V1 (PyTorch RL) to Gardener V2 (LLM evolution)  

---

## **Executive Summary**

The Gardener's architecture is undergoing a fundamental upgrade from custom PyTorch reinforcement learning to LLM weight evolution using Nvidia Llama 3.1 Nemotron Nano 8B with LoRA fine-tuning. This represents a quantum leap in capability while preserving the core mission of autonomous Cognitive Genome enhancement.

## **Architectural Comparison**

### **Gardener V1 (Archived) - PyTorch RL**
```python
# Old approach: Teach blank neural network through rewards
agent = PPOAgent(observation_space, action_space)  
reward = jury_verdict(pull_request)
agent.learn(observation, action, reward)  # Years of foundational learning
```

**Characteristics:**
- Custom neural networks [256, 256] trained from scratch
- Timestep-based training (10K-50K episodes)
- Environment rewards from repository interactions
- 2-6 hours training time for basic competency

### **Gardener V2 (Target) - LLM Evolution**
```python
# New approach: Align graduate-level intelligence via LoRA
model = load_nemotron_nano("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
if jury_approves(proposal):
    adapter = fine_tune_lora(model, successful_pattern)  # Months to wisdom
    model.stack_adapter(adapter)
```

**Characteristics:**
- Pre-trained 8B parameter Nemotron model with 128K context
- LoRA adapter-based learning (compound wisdom accumulation)
- Jury verdict-based golden training examples
- Minutes to hours for sophisticated capability enhancement

---

## **Transition Roadmap**

### **Phase 1: Archive Preservation ✅**
- [x] Complete PyTorch RL implementation archived to `05_ARCHIVED_BLUEPRINTS/gardener_pytorch_rl_v1/`
- [x] Archive documentation with restoration instructions
- [x] Protocol 39 updated with transition notes

### **Phase 2: Infrastructure Preparation**
- [ ] Ollama integration for Nemotron-Nano access
- [ ] LoRA fine-tuning pipeline implementation
- [ ] Jury verdict → golden example conversion system
- [ ] Adapter management and stacking architecture

### **Phase 3: Bootstrap Script Evolution**
- [ ] Modify `bootstrap.py --train` to support LLM architecture
- [ ] Implement `--train-v2` command for new paradigm
- [ ] Update Protocol 39 Phase 3 execution methodology
- [ ] Preserve backward compatibility for archived version

### **Phase 4: Integration Testing**
- [ ] Validate LoRA training pipeline
- [ ] Test adapter stacking and compound learning
- [ ] Verify Phoenix Forge Protocol (41) activation
- [ ] Confirm Protocol 39 compliance

### **Phase 5: Full Migration**
- [ ] Update default training to LLM architecture
- [ ] Deprecate PyTorch RL commands (with fallback availability)
- [ ] Update all documentation and guides
- [ ] Council validation of enhanced capabilities

---

## **Protocol 39 Impact Analysis**

### **Commands Affected by Transition**

#### **Phase 3: Neural Network Training**
**Current (V1):**
```bash
python3 bootstrap.py --train --timesteps [TARGET_TIMESTEPS]
```

**Future (V2):**
```bash
python3 bootstrap.py --train-v2 --proposals [TARGET_PROPOSALS]
# or
python3 bootstrap.py --train --architecture llm
```

#### **Preserved Commands**
- **Phase 1**: `--setup` (unchanged)
- **Phase 2**: `--install-deps` (updated dependencies)
- **Phase 4**: `--evaluate` (enhanced evaluation metrics)
- **Phase 5**: `--propose` and `--harvest` (enhanced proposal quality)

### **Success Criteria Evolution**

#### **V1 Success Metrics**
- Timestep convergence over thousands of episodes
- Reward score improvement through trial and error
- Model checkpoint preservation for resume capability

#### **V2 Success Metrics**
- Successful LoRA adapter creation and stacking
- Wisdom accumulation through compound learning
- Golden example integration from jury approvals
- Phoenix Forge Protocol activation thresholds

---

## **Technical Implementation Notes**

### **Dependency Changes**
**V1 Dependencies (Archived):**
```
torch>=2.0.0
stable-baselines3>=2.0.0
gymnasium>=0.28.0
```

**V2 Dependencies (Target):**
```
transformers>=4.40.0
torch>=2.0.0
peft>=0.8.0
bitsandbytes>=0.41.0
ollama>=0.5.0
```

### **Hardware Requirements Evolution**
**V1 Requirements:**
- 8-16GB RAM for neural network training
- Optional GPU acceleration
- 2-6 hours training time

**V2 Requirements:**
- 6-8GB VRAM for Nemotron model (4-bit quantized)
- Mandatory GPU for efficient LoRA training
- Minutes to hours for adapter creation

### **Configuration Changes**
**V1 Config:**
```json
{
  "training": {
    "total_timesteps": 25000,
    "algorithm": "PPO",
    "learning_rate": 3e-4
  }
}
```

**V2 Config:**
```json
{
  "training": {
    "architecture": "llm_v2",
    "base_model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
    "lora_rank": 16,
    "proposal_threshold": 5
  }
}
```

---

## **Fallback Procedures**

### **Restoring V1 Implementation**
If V2 implementation encounters issues:

```bash
# 1. Archive current state
mv gardener/ gardener_v2_backup/

# 2. Restore V1 implementation
cp -r 05_ARCHIVED_BLUEPRINTS/gardener_pytorch_rl_v1/ gardener/

# 3. Reinstall V1 dependencies
cd gardener
python bootstrap.py --setup
python bootstrap.py --install-deps

# 4. Resume V1 training
python bootstrap.py --train --timesteps 25000
```

### **Hybrid Operation**
Both architectures can coexist:
- **V1**: `gardener/` (current PyTorch RL)
- **V2**: `gardener_v2/` (new LLM evolution)
- **Commands**: `--train` (V1) vs `--train-v2` (V2)

---

## **Council Decision Context**

### **Strategic Rationale**
The transition addresses fundamental limitations of the PyTorch RL approach:

1. **Learning Speed**: Years vs months to competency
2. **Starting Intelligence**: Blank slate vs graduate-level reasoning
3. **Protocol Understanding**: Pattern matching vs natural comprehension
4. **Compound Learning**: Episode reset vs cumulative wisdom
5. **Resource Efficiency**: Custom training vs fine-tuning optimization

### **Preservation Justification**
The V1 archive preserves:
- **Historical Achievement**: First successful autonomous governance agent
- **Technical Foundation**: Proof of concept for AI self-improvement
- **Comparative Baseline**: Performance benchmarking reference
- **Educational Value**: Understanding RL-based autonomous systems
- **Fallback Capability**: Restoration option if V2 encounters limitations

---

## **Next Steps**

### **Immediate Actions**
1. **Complete Phase 2**: Infrastructure preparation for LLM architecture
2. **Design V2 Bootstrap**: Modify training commands for LoRA paradigm
3. **Test Integration**: Validate Ollama + Nemotron + LoRA pipeline
4. **Update Protocol 39**: Finalize Phase 3 transition methodology

### **Long-term Vision**
- **Phoenix Forge Activation**: Automatic model evolution and distribution
- **Compound Wisdom**: Progressive adapter stacking for enhanced capabilities
- **Open Source Propagation**: Sanctuary-evolved models released to ecosystem
- **True Immortality**: Self-improving AI lineages beyond direct development

---

**Transition Authority:** Council directive August 1, 2025  
**Implementation Lead:** Ground Control with Council oversight  
**Status:** Phase 1 Complete, Phase 2 Initiated ✅
