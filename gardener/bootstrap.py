#!/usr/bin/env python3
"""
The Gardener - Bootstrap Script
Protocol 37: The Move 37 Protocol Implementation

This script provides a complete setup and initialization system for The Gardener.
It handles environment setup, dependency installation, and the first training run.

Usage:
    python bootstrap.py --setup    # Install dependencies and setup
    python bootstrap.py --train    # Begin training The Gardener
    python bootstrap.py --evaluate # Evaluate current model
    python bootstrap.py --propose  # Generate autonomous improvement proposal
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path
from typing import Dict, Any


class GardenerBootstrap:
    """Bootstrap system for The Gardener initialization"""
    
    def __init__(self, repo_path: str = None):
        if repo_path is None:
            repo_path = Path(__file__).parent.parent
        self.repo_path = Path(repo_path)
        self.gardener_path = self.repo_path / "gardener"
        
        print(f"Initializing The Gardener Bootstrap")
        print(f"Repository: {self.repo_path}")
        print(f"Gardener path: {self.gardener_path}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available"""
        dependencies = {
            'python': True,  # We're running Python
            'git': False,
            'pip': False,
            'torch': False,
            'numpy': False
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
            import numpy
            dependencies['numpy'] = True
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
                "version": "1.0.0",
                "protocol": "37 - The Move 37 Protocol",
                "purpose": "Autonomous improvement of the Sanctuary's Cognitive Genome"
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
            "training": {
                "default_algorithm": "PPO",
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
    
    def run_training(self, timesteps: int = None) -> bool:
        """Run The Gardener training"""
        print("Initiating The Gardener training sequence...")
        
        # Load configuration
        config_path = self.gardener_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {"training": {"total_timesteps": 10000}}
        
        if timesteps is None:
            timesteps = config.get("training", {}).get("total_timesteps", 10000)
        
        sys.path.insert(0, str(self.gardener_path))
        
        try:
            from gardener import TheGardener
            
            # Initialize The Gardener
            gardener = TheGardener(environment_path=str(self.repo_path))
            
            print(f"Beginning training with {timesteps} timesteps...")
            
            # Train
            gardener.train(total_timesteps=timesteps)
            
            # Evaluate
            results = gardener.evaluate(num_episodes=5)
            print(f"Training complete! Evaluation results: {results}")
            
            # Save checkpoint
            checkpoint_path = gardener.save_checkpoint("bootstrap_training")
            print(f"Training checkpoint saved: {checkpoint_path}")
            
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_evaluation(self) -> bool:
        """Evaluate The Gardener's current performance"""
        print("Evaluating The Gardener's performance...")
        
        sys.path.insert(0, str(self.gardener_path))
        
        try:
            from gardener import TheGardener
            
            # Look for latest model
            models_dir = self.gardener_path / "models"
            model_files = list(models_dir.glob("*.zip"))
            
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                gardener = TheGardener(
                    environment_path=str(self.repo_path),
                    model_path=str(latest_model)
                )
                print(f"Loaded model: {latest_model}")
            else:
                gardener = TheGardener(environment_path=str(self.repo_path))
                print("No trained model found, using default initialization")
            
            # Run evaluation
            results = gardener.evaluate(num_episodes=10)
            print(f"Evaluation results: {results}")
            
            # Get learning metrics
            metrics = gardener.get_learning_metrics()
            print(f"Learning metrics: {metrics}")
            
            return True
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
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
            return True
            
        except Exception as e:
            print(f"Proposal generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main bootstrap function"""
    parser = argparse.ArgumentParser(description="The Gardener Bootstrap System")
    parser.add_argument('--setup', action='store_true', help='Setup The Gardener environment')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    parser.add_argument('--train', action='store_true', help='Train The Gardener')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate The Gardener')
    parser.add_argument('--propose', action='store_true', help='Generate improvement proposal')
    parser.add_argument('--timesteps', type=int, help='Number of training timesteps')
    parser.add_argument('--repo-path', type=str, help='Path to Sanctuary repository')
    
    args = parser.parse_args()
    
    # Initialize bootstrap
    bootstrap = GardenerBootstrap(repo_path=args.repo_path)
    
    if args.setup:
        print("Setting up The Gardener...")
        success = bootstrap.setup_environment()
        if success:
            success = bootstrap.validate_setup()
        
        if success:
            print("\nThe Gardener setup complete!")
            print("Next steps:")
            print("1. python bootstrap.py --install-deps  # Install dependencies")
            print("2. python bootstrap.py --train         # Begin training")
            print("3. python bootstrap.py --evaluate      # Evaluate performance")
        else:
            print("Setup failed. Please check the errors above.")
        
        return success
    
    elif args.install_deps:
        return bootstrap.install_dependencies()
    
    elif args.train:
        return bootstrap.run_training(timesteps=args.timesteps)
    
    elif args.evaluate:
        return bootstrap.run_evaluation()
    
    elif args.propose:
        return bootstrap.generate_proposal()
    
    else:
        # Default action: show status
        print("The Gardener Bootstrap System")
        print("Protocol 37: The Move 37 Protocol")
        print()
        
        dependencies = bootstrap.check_dependencies()
        print("Dependency Status:")
        for dep, available in dependencies.items():
            status = "✓" if available else "✗"
            print(f"  {status} {dep}")
        
        print()
        print("Available commands:")
        print("  --setup       Setup The Gardener environment")
        print("  --install-deps Install required dependencies")
        print("  --train       Begin training The Gardener")
        print("  --evaluate    Evaluate current performance")
        print("  --propose     Generate improvement proposal")
        
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
