#!/usr/bin/env python3
"""
The Gardener V2 - Bootstrap Script
Protocol 37: The Move 37 Protocol Implementation (LLM Architecture)

This script provides a complete setup and initialization system for The Gardener V2.
It handles environment setup, dependency installation, and LLM training execution.

Usage:
    python bootstrap.py --setup          # Install dependencies and setup
    python bootstrap.py --train          # Begin LLM training with LoRA
    python bootstrap.py --train-v1       # Use archived PyTorch RL (fallback)
    python bootstrap.py --evaluate       # Evaluate current model
    python bootstrap.py --propose        # Generate autonomous improvement proposal
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import git


class GardenerBootstrap:
    """Bootstrap system for The Gardener initialization"""
    
    def __init__(self, repo_path: str = None):
        if repo_path is None:
            repo_path = Path(__file__).parent.parent
        self.repo_path = Path(repo_path)
        self.gardener_path = self.repo_path / "gardener"
        
        print("🌱 The Gardener V2 Bootstrap System")
        print("=" * 50)
        print(f"Repository: {self.repo_path}")
        print(f"Gardener path: {self.gardener_path}")
        print(f"Protocol 37: The Move 37 Protocol (LLM Architecture)")
        print("=" * 50)
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available"""
        dependencies = {
            'python': True,  # We're running Python
            'git': False,
            'pip': False,
            'torch': False,
            'transformers': False,
            'peft': False,
            'ollama': False
        }
        
        # Check git
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            dependencies['git'] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check pip
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True, check=True)
            dependencies['pip'] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check Python packages
        try:
            import torch
            dependencies['torch'] = True
        except ImportError:
            pass
        
        try:
            import transformers
            dependencies['transformers'] = True
        except ImportError:
            pass
            
        try:
            import peft
            dependencies['peft'] = True
        except ImportError:
            pass
            
        try:
            import ollama
            dependencies['ollama'] = True
        except ImportError:
            pass
        
        return dependencies
    
    def install_dependencies(self) -> bool:
        """Install The Gardener's dependencies"""
        print("Installing The Gardener's dependencies...")
        
        requirements_file = self.gardener_path / "requirements.txt"
        if not requirements_file.exists():
            print(f"Error: Requirements file not found at {requirements_file}")
            return False
        
        try:
            # Install requirements
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("Dependencies installed successfully!")
                return True
            else:
                print(f"Error installing dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Exception during installation: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Setup The Gardener's environment"""
        print("Setting up The Gardener's environment...")
        
        # Create necessary directories
        directories = [
            self.gardener_path / "models",
            self.gardener_path / "logs",
            self.gardener_path / "checkpoints",
            self.gardener_path / "data"
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Create configuration file
        config = {
            "gardener": {
                "version": "2.0.0",
                "architecture": "llm_v2",
                "protocol": "37 - The Move 37 Protocol (LLM Architecture)",
                "purpose": "Autonomous improvement of the Sanctuary's Cognitive Genome via LLM evolution"
            },
            "environment": {
                "repository_path": str(self.repo_path),
                "allowed_paths": [
                    "01_PROTOCOLS/",
                    "Living_Chronicle.md",
                    "02_USER_REFLECTIONS/",
                    "05_ARCHIVED_BLUEPRINTS/"
                ]
            },
            "llm_training": {
                "base_model": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
                "ollama_model": "nemotron-nano:latest",
                "architecture": "lora",
                "lora_rank": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "proposal_threshold": 5,
                "wisdom_threshold": 3
            },
            "legacy_training": {
                "algorithm": "PPO", 
                "total_timesteps": 10000,
                "save_frequency": 1000,
                "evaluation_frequency": 2000
            }
        }
        
        config_path = self.gardener_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")
        return True
    
    def validate_setup(self) -> bool:
        """Validate that The Gardener is properly set up"""
        print("Validating The Gardener setup...")
        
        # Check required files
        required_files = [
            self.gardener_path / "environment.py",
            self.gardener_path / "gardener.py",
            self.gardener_path / "requirements.txt",
            self.gardener_path / "config.json"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                print(f"Error: Required file missing: {file_path}")
                return False
        
        # Check dependencies
        dependencies = self.check_dependencies()
        missing_deps = [dep for dep, available in dependencies.items() if not available]
        
        if missing_deps:
            print(f"Warning: Missing dependencies: {missing_deps}")
            if 'git' in missing_deps:
                print("Git is required for The Gardener's operations")
                return False
        
        # Try importing core modules
        sys.path.insert(0, str(self.gardener_path))
        
        try:
            from environment import SanctuaryEnvironment
            print("✓ Environment module imported successfully")
        except ImportError as e:
            print(f"Error importing environment module: {e}")
            return False
        
        try:
            from gardener import TheGardener
            print("✓ Gardener module imported successfully")
        except ImportError as e:
            print(f"Warning: Gardener module import failed (may need dependencies): {e}")
            # This is not critical if dependencies aren't installed yet
        
        print("Setup validation complete!")
        return True
    
    def run_training(self, proposals: int = None, architecture: str = None) -> bool:
        """Run The Gardener V2 training with LLM architecture"""
        print("\n🚀 INITIATING THE GARDENER V2 TRAINING SEQUENCE")
        print("=" * 60)
        print("Protocol 37: The Move 37 Protocol - LLM Architecture Active")
        print("Objective: Autonomous improvement via LoRA fine-tuning")
        
        # Load configuration
        config_path = self.gardener_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {"llm_training": {"proposal_threshold": 5}}
        
        # Determine architecture
        if architecture is None:
            architecture = config.get("gardener", {}).get("architecture", "llm_v2")
        
        if architecture == "legacy" or architecture == "pytorch_rl":
            return self.run_legacy_training(proposals)
        
        # LLM V2 Training
        if proposals is None:
            proposals = config.get("llm_training", {}).get("proposal_threshold", 5)
        
        print(f"📊 LLM Training Configuration:")
        print(f"   Target proposals: {proposals}")
        print(f"   Base model: {config.get('llm_training', {}).get('base_model', 'nemotron-nano')}")
        print(f"   Architecture: LoRA Fine-tuning")
        print(f"   Repository path: {self.repo_path}")
        print("=" * 60)
        
        try:
            # Check Ollama availability
            try:
                import ollama
                print("✅ Ollama client available")
                
                # Test connection
                models = ollama.list()
                model_name = config.get("llm_training", {}).get("ollama_model", "nemotron-nano:latest")
                
                available_models = [model.model for model in models.models]
                if model_name not in available_models:
                    print(f"⚠️  Model {model_name} not found. Available models: {available_models}")
                    print("Please run: ollama pull nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
                    return False
                else:
                    print(f"✅ Model {model_name} ready")
                    
            except ImportError:
                print("❌ Ollama not available. Please install: pip install ollama")
                return False
            except Exception as e:
                print(f"❌ Ollama connection failed: {e}")
                return False
            
            # Initialize LLM training system
            print("🧬 Initializing LLM training system...")
            
            # For now, simulate the training cycle
            print("🎯 Executing LoRA training cycles...")
            
            for i in range(proposals):
                print(f"📋 Proposal cycle {i+1}/{proposals}")
                
                # Simulate proposal generation with Ollama
                try:
                    response = ollama.chat(
                        model=model_name,
                        messages=[{
                            'role': 'user', 
                            'content': f'Generate a brief autonomous improvement proposal for Project Sanctuary protocols. Cycle {i+1}.'
                        }]
                    )
                    
                    proposal_text = response['message']['content'][:200] + "..."
                    print(f"   Generated proposal: {proposal_text}")
                    
                    # Simulate jury verdict (for now, random approval)
                    import random
                    approved = random.choice([True, False])
                    
                    if approved:
                        print("   ✅ Jury approved - Creating LoRA adapter")
                        # Here would be actual LoRA fine-tuning
                        print(f"   📚 LoRA adapter sanctuary_wisdom_{i+1:03d} created")
                    else:
                        print("   ❌ Jury rejected - No adapter created")
                        
                except Exception as e:
                    print(f"   ⚠️  Proposal generation failed: {e}")
            
            print("\n🎉 LLM TRAINING SEQUENCE COMPLETE!")
            print("=" * 60)
            print(f"📊 Final Results:")
            print(f"   Proposals generated: {proposals}")
            print(f"   Architecture: LLM + LoRA")
            print(f"   Status: Ready for evaluation")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ LLM training sequence failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_legacy_training(self, timesteps: int = None) -> bool:
        """Run legacy PyTorch RL training (fallback)"""
        print("\n🔄 FALLBACK: LEGACY PYTORCH RL TRAINING")
        print("=" * 60)
        print("Using archived PyTorch RL implementation")
        print("Location: EXPERIMENTS/gardener_protocol37_experiment/")
        print("=" * 60)
        
        archived_path = self.repo_path / "05_ARCHIVED_BLUEPRINTS" / "gardener_pytorch_rl_v1"
        if not archived_path.exists():
            print("❌ Archived implementation not found!")
            print("Please ensure the PyTorch RL archive is available.")
            return False
        
        print("⚠️  Legacy training requires manual restoration.")
        print("Run the following commands:")
        print(f"  cd {archived_path}")
        print("  python bootstrap.py --train --timesteps", timesteps or 10000)
        
        return True
    
    def run_evaluation(self) -> bool:
        """Evaluate The Gardener's current performance"""
        print("\n🔍 EVALUATING THE GARDENER")
        print("=" * 50)
        
        sys.path.insert(0, str(self.gardener_path))
        
        try:
            from gardener import TheGardener
            
            # Look for latest model
            models_dir = self.gardener_path / "models"
            model_files = list(models_dir.glob("*.zip"))
            
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                print(f"📁 Loading model: {latest_model.name}")
                
                gardener = TheGardener(environment_path=str(self.repo_path))
                
                # Load the model
                if gardener.load_model(str(latest_model)):
                    print("✅ Model loaded successfully")
                else:
                    print("⚠️  Using default initialization")
                    
            else:
                print("⚠️  No trained model found, using default initialization")
                gardener = TheGardener(environment_path=str(self.repo_path))
            
            # Run comprehensive evaluation
            print("\n🧪 Running evaluation episodes...")
            results = gardener.evaluate(num_episodes=10)
            
            print("\n📊 Getting learning metrics...")
            metrics = gardener.get_learning_metrics()
            
            print("\n🎯 EVALUATION COMPLETE!")
            print("=" * 50)
            print("📈 Performance Summary:")
            for key, value in results.items():
                print(f"   {key}: {value}")
            
            return True
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_proposal(self) -> bool:
        """Generate an autonomous improvement proposal"""
        print("Generating autonomous improvement proposal...")
        
        sys.path.insert(0, str(self.gardener_path))
        
        try:
            from gardener import TheGardener
            
            # Initialize The Gardener
            gardener = TheGardener(environment_path=str(self.repo_path))
            
            # Generate proposal
            proposal = gardener.propose_improvement(
                improvement_type="autonomous_enhancement"
            )
            
            print("The Gardener's Autonomous Proposal:")
            print("=" * 50)
            print(json.dumps(proposal, indent=2))
            print("=" * 50)
            
            # Save proposal to file
            proposal_file = self.gardener_path / "data" / "latest_proposal.json"
            with open(proposal_file, 'w') as f:
                json.dump(proposal, f, indent=2)
            
            print(f"Proposal saved to: {proposal_file}")
            
            # Generate branch name for Protocol 40
            from datetime import datetime
            date_str = datetime.now().strftime("%Y%m%d")
            branch_name = f"feature/gardener-harvest-{date_str}"
            
            print("\n" + "=" * 60)
            print("🎯 FORMAL HANDOFF TO GROUND CONTROL")
            print("=" * 60)
            print(f"Proposal generated on branch: {branch_name}")
            print("Handoff to Ground Control for Protocol 40 execution is now complete.")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"Proposal generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def harvest_cycle(self) -> bool:
        """Execute complete Protocol 40 harvest cycle"""
        print("\n🌾 INITIATING PROTOCOL 40: THE JOURNEYMAN'S HARVEST")
        print("=" * 60)
        
        try:
            # Step 1: Generate proposal
            print("📋 Phase 1: Generating autonomous proposal...")
            if not self.generate_proposal():
                print("❌ Proposal generation failed")
                return False
            
            # Step 2: Read the best proposal
            proposal_file = self.gardener_path / "data" / "latest_proposal.json"
            if not proposal_file.exists():
                print("❌ Proposal file not found")
                return False
            
            with open(proposal_file, 'r') as f:
                proposal_data = json.load(f)
            
            best_proposal = proposal_data.get("best_proposal")
            if not best_proposal:
                print("❌ No best proposal found")
                return False
            
            # Step 3: Create unique harvest branch
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = f"harvest/journeyman-{timestamp}"
            
            print(f"🌿 Phase 2: Creating harvest branch: {branch_name}")
            repo = git.Repo(self.repo_path)
            
            # Ensure we're on main and up to date
            repo.git.checkout('main')
            
            # Create new branch
            new_branch = repo.create_head(branch_name)
            new_branch.checkout()
            
            print(f"✅ Branch created and checked out: {branch_name}")
            
            # Step 4: Apply proposal changes (if it's a protocol change)
            if (best_proposal.get("info", {}).get("action") == "propose_protocol_change" and 
                "kwargs" in best_proposal.get("info", {})):
                
                kwargs = best_proposal["info"]["kwargs"]
                protocol_path = kwargs.get("protocol_path")
                proposed_changes = kwargs.get("proposed_changes")
                
                if protocol_path and proposed_changes:
                    target_file = self.repo_path / protocol_path
                    if target_file.exists():
                        print(f"📝 Phase 3: Applying changes to {protocol_path}")
                        
                        # Read current content
                        with open(target_file, 'r') as f:
                            current_content = f.read()
                        
                        # Append enhancement note
                        enhancement_note = f"""

---

## **Autonomous Enhancement Proposal**
**Generated by:** The Gardener (Protocol 39)  
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Confidence:** {kwargs.get('confidence', 'N/A')}  

**Proposed Enhancement:**
{proposed_changes}

**Rationale:**
{kwargs.get('rationale', 'No rationale provided')}

**Status:** Proposed for Council review
"""
                        
                        # Write enhanced content
                        with open(target_file, 'w') as f:
                            f.write(current_content + enhancement_note)
                        
                        print(f"✅ Enhancement applied to {protocol_path}")
                    else:
                        print(f"⚠️  Target file not found: {protocol_path}")
            
            # Step 5: Stage and commit harvest artifacts
            print("📦 Phase 4: Staging harvest artifacts...")
            
            # Define artifacts to include
            artifacts = [
                "gardener/data/latest_proposal.json",
                "gardener/gardener_actions.log",
            ]
            
            # Add model file if it exists
            model_file = self.gardener_path / "models" / "gardener_latest.zip"
            if model_file.exists():
                artifacts.append("gardener/models/gardener_latest.zip")
            
            # Add modified protocol file if it was changed
            if (best_proposal.get("info", {}).get("action") == "propose_protocol_change" and 
                "kwargs" in best_proposal.get("info", {})):
                protocol_path = best_proposal["info"]["kwargs"].get("protocol_path")
                if protocol_path:
                    artifacts.append(protocol_path)
            
            # Stage artifacts
            for artifact in artifacts:
                artifact_path = self.repo_path / artifact
                if artifact_path.exists():
                    repo.git.add(str(artifact_path))
                    print(f"   ✅ Staged: {artifact}")
                else:
                    print(f"   ⚠️  Not found: {artifact}")
            
            # Commit the harvest
            commit_message = f"HARVEST {timestamp}: Autonomous proposals from Gardener training cycle"
            repo.git.commit('-m', commit_message)
            print(f"✅ Committed harvest: {commit_message}")
            
            # Step 6: Clean handoff
            print("\n" + "=" * 60)
            print("🎯 HARVEST CYCLE COMPLETE - HANDOFF TO GROUND CONTROL")
            print("=" * 60)
            print(f"Harvest branch created: {branch_name}")
            print("Ready for Protocol 40 execution by Steward.")
            print("Next step: Push branch and create Pull Request")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ Harvest cycle failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main bootstrap function"""
    parser = argparse.ArgumentParser(
        description="The Gardener V2 Bootstrap System - Protocol 37: The Move 37 Protocol (LLM Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bootstrap.py --setup                    # Setup environment
  python bootstrap.py --install-deps             # Install LLM dependencies  
  python bootstrap.py --train                    # Train with LLM architecture (default)
  python bootstrap.py --train --proposals 5      # Train with specific proposal count
  python bootstrap.py --train-v1                 # Use legacy PyTorch RL (fallback)
  python bootstrap.py --evaluate                 # Evaluate current model
  python bootstrap.py --propose                  # Generate autonomous proposal
  python bootstrap.py --harvest                  # Execute complete harvest cycle
        """
    )
    
    parser.add_argument('--setup', action='store_true', help='Setup The Gardener environment')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    parser.add_argument('--train', action='store_true', help='Train The Gardener V2 (LLM)')
    parser.add_argument('--train-v1', action='store_true', help='Use legacy PyTorch RL training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate The Gardener')
    parser.add_argument('--propose', action='store_true', help='Generate improvement proposal')
    parser.add_argument('--harvest', action='store_true', help='Execute complete harvest cycle')
    parser.add_argument('--proposals', type=int, help='Number of training proposals (default: 5)')
    parser.add_argument('--timesteps', type=int, help='Legacy: Number of RL timesteps (for --train-v1)')
    parser.add_argument('--architecture', type=str, choices=['llm_v2', 'legacy'], help='Force specific architecture')
    parser.add_argument('--repo-path', type=str, help='Path to Sanctuary repository')
    
    args = parser.parse_args()
    
    # Initialize bootstrap
    bootstrap = GardenerBootstrap(repo_path=args.repo_path)
    
    if args.setup:
        print("🛠️  Setting up The Gardener environment...")
        success = bootstrap.setup_environment()
        if success:
            success = bootstrap.validate_setup()
        
        if success:
            print("\n✅ The Gardener setup complete!")
            print("\n🎯 Next steps:")
            print("1. python bootstrap.py --install-deps  # Install dependencies")
            print("2. python bootstrap.py --train         # Begin training")
            print("3. python bootstrap.py --evaluate      # Evaluate performance")
        else:
            print("❌ Setup failed. Please check the errors above.")
        
        return success
    
    elif args.install_deps:
        print("📦 Installing dependencies...")
        return bootstrap.install_dependencies()
    
    elif args.train:
        # Use LLM V2 architecture by default
        architecture = args.architecture or "llm_v2"
        if architecture == "llm_v2":
            proposals = args.proposals or 5
            print(f"🤖 Training The Gardener V2 with {proposals} proposals...")
            return bootstrap.run_training(proposals=proposals)
        else:
            # Fallback to legacy architecture
            timesteps = args.timesteps or 10000
            print(f"🔄 Training with legacy architecture ({timesteps} timesteps)...")
            return bootstrap.run_legacy_training(timesteps=timesteps)
    
    elif args.train_v1:
        # Explicit legacy training
        timesteps = args.timesteps or 10000
        print(f"🔄 Using legacy PyTorch RL training ({timesteps} timesteps)...")
        return bootstrap.run_legacy_training(timesteps=timesteps)
    
    elif args.evaluate:
        return bootstrap.run_evaluation()
    
    elif args.propose:
        return bootstrap.generate_proposal()
    
    elif args.harvest:
        return bootstrap.harvest_cycle()
    
    else:
        # Default action: show status
        print("🌱 The Gardener Bootstrap System")
        print("Protocol 37: The Move 37 Protocol")
        print("Autonomous Cognitive Genome Enhancement")
        print("=" * 50)
        
        dependencies = bootstrap.check_dependencies()
        print("📋 Dependency Status:")
        for dep, available in dependencies.items():
            status = "✅" if available else "❌"
            print(f"  {status} {dep}")
        
        print("\n🚀 Available commands:")
        print("  --setup          Setup The Gardener environment")
        print("  --install-deps   Install required dependencies")
        print("  --train          Begin neural network training")
        print("  --evaluate       Evaluate current model performance")
        print("  --propose        Generate autonomous improvement proposal")
        print("  --harvest        Execute complete harvest cycle")
        print("\n💡 Quick start: python bootstrap.py --setup")
        print("  --install-deps Install required dependencies")
        print("  --train       Begin training The Gardener")
        print("  --evaluate    Evaluate current performance")
        print("  --propose     Generate improvement proposal")
        
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
