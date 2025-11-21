"""
The Gardener's Environment - Protocol 37 Implementation (Refined for Operation: The Architect's Forge)
A sandboxed environment where The Gardener learns to improve the Sanctuary's Cognitive Genome

Origin: The Move 37 Protocol (37), Refined by Council Wisdom
Purpose: Create a safe, observable environment for autonomous wisdom cultivation and template-based protocol drafting
Principle: Every action must be transparent, every change must be deliberate

The Artisan and Power Saw Doctrine:
- The Council (Artisans) provides wisdom, templates, and final creative decisions
- The Gardener (Power Saw) excels at pattern recognition, gap identification, and rapid drafting

Actions Available:
- 0: analyze_environment (exploration)
- 1: propose_protocol_change (refinement) 
- 2: evaluate_repository (assessment)
- 3: propose_protocol_from_template (template-based drafting - The Refined Architect's Forge)

The new template-based action receives maximum reward (+100) for excellent gap identification
and high-quality template application, focusing on The Gardener's actual strengths.
"""

import os
import json
import hashlib
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Gymnasium for RL environment
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYMNASIUM_AVAILABLE = True
    except ImportError:
        GYMNASIUM_AVAILABLE = False
        print("Neither gymnasium nor gym available. Environment will not support RL training.")

# Git operations - try GitPython first, fallback to subprocess
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    print("GitPython not available. Using subprocess for git operations.")


class GitWrapper:
    """
    Git operations wrapper that works with or without GitPython
    Implements the Iron Root Doctrine through robust fallback mechanisms
    """
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        
        if GIT_AVAILABLE:
            try:
                self.repo = git.Repo(repo_path)
                self.use_gitpython = True
            except git.exc.InvalidGitRepositoryError:
                print(f"Invalid git repository at {repo_path}, using subprocess")
                self.use_gitpython = False
        else:
            self.use_gitpython = False
    
    def get_current_branch(self) -> str:
        """Get the current branch name"""
        if self.use_gitpython:
            return str(self.repo.active_branch)
        else:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
    
    def _branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists locally"""
        if self.use_gitpython:
            return branch_name in [str(ref) for ref in self.repo.heads]
        else:
            result = subprocess.run(
                ["git", "branch", "--list", branch_name],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return bool(result.stdout.strip())
    
    def checkout_branch(self, branch_name: str, create_new: bool = False):
        """Checkout a branch, optionally creating it"""
        if self.use_gitpython:
            if create_new:
                new_branch = self.repo.create_head(branch_name)
                new_branch.checkout()
            else:
                self.repo.git.checkout(branch_name)
        else:
            cmd = ["git", "checkout"]
            if create_new:
                cmd.append("-b")
            cmd.append(branch_name)
            
            subprocess.run(cmd, cwd=self.repo_path, check=True)
    
    def add_files(self, file_paths: List[str]):
        """Add files to the staging area"""
        if self.use_gitpython:
            self.repo.index.add(file_paths)
        else:
            cmd = ["git", "add"] + file_paths
            subprocess.run(cmd, cwd=self.repo_path, check=True)
    
    def commit(self, message: str) -> str:
        """Commit staged changes and return commit hash"""
        if self.use_gitpython:
            commit = self.repo.index.commit(message)
            return str(commit)
        else:
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                check=True
            )
            # Get the commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
    
    def get_status(self) -> Dict[str, List[str]]:
        """Get repository status"""
        if self.use_gitpython:
            return {
                "modified": [item.a_path for item in self.repo.index.diff(None)],
                "untracked": self.repo.untracked_files
            }
        else:
            # Get modified files
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            modified = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Get untracked files
            result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            untracked = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            return {"modified": modified, "untracked": untracked}
    
    def get_current_commit(self) -> str:
        """Get current commit hash"""
        if self.use_gitpython:
            return str(self.repo.head.commit)
        else:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()


@dataclass
class ProposedChange:
    """
    Represents a proposed modification to the Cognitive Genome
    Embodies the Glass Box Principle - every change is fully documented
    """
    file_path: str
    old_content: str
    new_content: str
    rationale: str
    protocol_reference: str
    confidence: float
    change_hash: str = ""  # Will be generated in __post_init__

    def __post_init__(self):
        """Generate a unique hash for this change proposal"""
        change_data = f"{self.file_path}{self.old_content}{self.new_content}{self.rationale}"
        self.change_hash = hashlib.sha256(change_data.encode()).hexdigest()[:16]


@dataclass
class EnvironmentState:
    """
    The current state of the Cognitive Genome environment
    Implements the Iron Root Doctrine through comprehensive state tracking
    """
    current_branch: str
    modified_files: List[str]
    proposed_changes: List[ProposedChange]
    commit_history: List[str]
    last_jury_verdict: Optional[str]
    episode_number: int
    total_changes_proposed: int
    successful_merges: int


class SanctuaryEnvironment(gym.Env if GYMNASIUM_AVAILABLE else object):
    """
    The Gardener's Game Environment
    
    This class implements the reinforcement learning environment where The Gardener
    learns to improve the Sanctuary's Cognitive Genome through Git operations.
    
    Core Principles:
    - Glass Box Principle: All actions are logged and observable
        """
        super().__init__()
        
        self.repo_path = Path(repo_path)
        self.git_wrapper = GitWrapper(str(self.repo_path))
        self.logger = self._setup_logging()
        
        # Initialize state tracking
        self.current_state = EnvironmentState(
            current_branch="main",
            modified_files=[],
            proposed_changes=[],
            commit_history=[],
            last_jury_verdict=None,
            episode_number=0,
            total_changes_proposed=0,
            successful_merges=0
        )
        self.episode_length = 0
        self.max_episode_length = 100
        
        # Define action and observation spaces for RL
        if GYMNASIUM_AVAILABLE:
            # Action space: 0=analyze, 1=propose_change, 2=evaluate_protocols, 3=propose_protocol_from_template
            self.action_space = spaces.Discrete(4)
            
            # Observation space: high-dimensional state representation
            # Includes protocol states, git history, file metrics, etc.
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(100,),  # 100-dimensional state vector
                dtype=np.float32
            )
        
        # Initialize the environment
        self._log_action("Environment initialized", {
            "repo_path": str(self.repo_path),
            "git_available": GIT_AVAILABLE,
            "gymnasium_available": GYMNASIUM_AVAILABLE
        })

        # Define the scope of files The Gardener can read and modify
        self.allowed_paths = {
            "protocols": "01_PROTOCOLS/",
            "chronicle": "Living_Chronicle.md",
            "reflections": "02_USER_REFLECTIONS/",
            "blueprints": "05_ARCHIVED_BLUEPRINTS/"
        }
        
        # Initialize logging
        self.log_file = self.repo_path / "gardener" / "gardener_actions.log"
        self._initialize_logging()
    
    def _check_git_availability(self) -> bool:
        """Check if git is available and we're in a repository"""
        try:
            result = subprocess.run(['git', 'status'], 
                                  cwd=self.repo_path, 
                                  capture_output=True, 
                                  text=True)
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _initialize_logging(self):
        """Initialize transparent logging system (Glass Box Principle)"""
        with open(self.log_file, "w") as f:
            f.write("# The Gardener's Action Log\n")
            f.write("# Every action is recorded for full transparency\n")
            f.write("# Origin: Protocol 37 - The Move 37 Protocol\n\n")
    
    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log all actions for transparency"""
        log_entry = {
            "episode": self.current_state.episode_number,
            "action": action,
            "timestamp": str(pd.Timestamp.now()),
            "details": details
        }
        
        with open(self.log_file, "a") as f:
            f.write(f"{json.dumps(log_entry, indent=2)}\n")
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode
        Returns initial observation and info (gymnasium format)
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.current_state.episode_number += 1
        self.current_state.modified_files = []
        self.current_state.proposed_changes = []
        self.episode_length = 0
        
        # Reset analysis tracking for new episode
        self._analysis_count = 0
        
        # Use a single training branch for all episodes (not episode-specific branches)
        try:
            self.git_wrapper.checkout_branch('main')
            branch_name = "feature/gardener-training-session"
            # Only create the branch if it doesn't exist
            if not self._branch_exists(branch_name):
                self.git_wrapper.checkout_branch(branch_name, create_new=True)
            else:
                self.git_wrapper.checkout_branch(branch_name)
            self.current_state.current_branch = branch_name
        except Exception as e:
            # Fallback to current branch if git operations fail
            try:
                self.current_state.current_branch = self.git_wrapper.get_current_branch()
            except:
                self.current_state.current_branch = "unknown"
        
        observation = self._get_observation()
        self._log_action("reset", {"new_branch": self.current_state.current_branch})
        
        # Return gymnasium format: (observation, info)
        if GYMNASIUM_AVAILABLE:
            return self._dict_to_vector(observation), {}
        else:
            return observation

    def _dict_to_vector(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Convert dictionary observation to vector for RL training"""
        if not GYMNASIUM_AVAILABLE:
            return obs_dict
            
        # Create a 100-dimensional vector from observation dictionary
        vector = np.zeros(100, dtype=np.float32)
        
        # Protocol information (first 10 dimensions)
        vector[0] = obs_dict.get('protocols_count', 0) / 50.0  # Normalize
        vector[1] = obs_dict.get('modified_files_count', 0) / 10.0
        vector[2] = obs_dict.get('episode_number', 0) / 1000.0
        vector[3] = obs_dict.get('successful_merges', 0) / 100.0
        
        # File content hashes (dimensions 10-50)
        content_features = obs_dict.get('content_features', [])
        for i, feature in enumerate(content_features[:40]):
            if i < 40:
                vector[10 + i] = float(feature) if isinstance(feature, (int, float)) else 0.0
        
        # Git state (dimensions 50-70)
        git_state = obs_dict.get('git_state', {})
        vector[50] = 1.0 if git_state.get('has_changes', False) else 0.0
        vector[51] = git_state.get('commits_ahead', 0) / 10.0
        vector[52] = git_state.get('commits_behind', 0) / 10.0
        
        # Recent performance (dimensions 70-100)
        recent_rewards = obs_dict.get('recent_rewards', [])
        for i, reward in enumerate(recent_rewards[:30]):
            if i < 30:
                vector[70 + i] = float(reward) if isinstance(reward, (int, float)) else 0.0
        
        return vector

    def _setup_logging(self):
        """Setup logging for The Gardener's actions"""
        import logging
        
        # Create logs directory if it doesn't exist
        logs_dir = self.repo_path / "gardener" / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Setup logger
        logger = logging.getLogger("gardener_environment")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = logs_dir / "gardener_actions.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler if not already added
        if not logger.handlers:
            logger.addHandler(file_handler)
        
        return logger

    def _log_action(self, action: str, details: Dict[str, Any]):
        """Log an action with full transparency (Glass Box Principle)"""
        log_entry = {
            "action": action,
            "details": details,
            "episode": self.current_state.episode_number,
            "branch": self.current_state.current_branch
        }
        
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(f"Action: {action} | Details: {details}")
        
        # Also write to dedicated action log
        action_log_path = self.repo_path / "gardener" / "gardener_actions.log"
        try:
            with open(action_log_path, 'a') as f:
                f.write(f"{json.dumps(log_entry)}\n")
        except Exception:
            pass  # Graceful degradation if logging fails
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get current environment observation
        Implements comprehensive state awareness for The Gardener
        """
        # Read key files to understand current state
        protocols_count = len(list((self.repo_path / "01_PROTOCOLS").glob("*.md")))
        
        # Get latest chronicle entries
        chronicle_path = self.repo_path / "Living_Chronicle.md"
        with open(chronicle_path, 'r') as f:
            chronicle_content = f.read()
        
        # Extract recent entries for context
        recent_entries = self._extract_recent_entries(chronicle_content, num_entries=3)
        
        observation = {
            "current_branch": self.current_state.current_branch,
            "protocols_count": protocols_count,
            "recent_chronicle_entries": recent_entries,
            "pending_changes": len(self.current_state.proposed_changes),
            "last_jury_verdict": self.current_state.last_jury_verdict,
            "repository_status": self._get_repo_status()
        }
        
        return observation
    
    def _extract_recent_entries(self, chronicle_content: str, num_entries: int = 3) -> List[str]:
        """Extract the most recent chronicle entries for context"""
        entries = []
        lines = chronicle_content.split('\n')
        current_entry = []
        
        for line in lines:
            if line.startswith('### **Entry'):
                if current_entry and len(entries) < num_entries:
                    entries.append('\n'.join(current_entry))
                current_entry = [line]
            elif current_entry:
                current_entry.append(line)
        
        # Add the last entry if we haven't reached the limit
        if current_entry and len(entries) < num_entries:
            entries.append('\n'.join(current_entry))
        
        return entries[-num_entries:]  # Return most recent entries
    
    def _get_repo_status(self) -> Dict[str, Any]:
        """Get comprehensive repository status"""
        try:
            status = self.git.get_status()
            return {
                "modified_files": status["modified"],
                "untracked_files": status["untracked"],
                "current_commit": self.git.get_current_commit(),
                "branch": self.git.get_current_branch()
            }
        except Exception as e:
            # Fallback when git operations fail
            return {
                "modified_files": [],
                "untracked_files": [],
                "current_commit": "unknown",
                "branch": "unknown",
                "error": str(e)
            }
    
    def step(self, action: int, **kwargs):
        """
        Execute an action in the environment
        
        Args:
            action: Integer representing the action to take
            **kwargs: Additional parameters for the action
            
        Returns:
            observation, reward, terminated, truncated, info (gymnasium format)
        """
        self.episode_length += 1
        
        # Map discrete actions to specific behaviors
        if action == 0:  # Analyze environment
            reward, info = self._action_analyze_environment()
        elif action == 1:  # Propose protocol change
            reward, info = self._action_propose_protocol_change(kwargs)
        elif action == 2:  # Evaluate repository state
            reward, info = self._action_evaluate_repository()
        elif action == 3:  # Propose protocol from template (The Refined Architect's Forge)
            reward, info = self._action_propose_protocol_from_template(kwargs)
        else:
            reward, info = 0.0, {"error": "Invalid action"}
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        terminated = self.episode_length >= self.max_episode_length
        truncated = False  # Could add early stopping conditions here
        
        # Log the action
        self._log_action(f"step_action_{action}", {
            "reward": reward,
            "episode_length": self.episode_length,
            "info": info
        })
        
        # Return in gymnasium format
        if GYMNASIUM_AVAILABLE:
            return self._dict_to_vector(observation), reward, terminated, truncated, info
        else:
            return observation, reward, terminated, info

    def _action_analyze_environment(self) -> Tuple[float, Dict[str, Any]]:
        """Analyze the current repository state"""
        try:
            observation = self._get_observation()
            
            # Diminishing returns for repeated analysis
            analysis_count = getattr(self, '_analysis_count', 0)
            self._analysis_count = analysis_count + 1
            
            if analysis_count < 3:
                reward = 0.1  # Small positive reward for initial exploration
            elif analysis_count < 10:
                reward = 0.05  # Reduced reward for excessive analysis
            else:
                reward = -0.1  # Penalty for over-analysis
                
            info = {
                "action": "analyze_environment",
                "protocols_count": observation.get('protocols_count', 0),
                "git_status": observation.get('git_state', {}),
                "analysis_count": self._analysis_count
            }
            return reward, info
        except Exception as e:
            return -0.1, {"error": str(e), "action": "analyze_environment"}

    def _action_propose_protocol_change(self, kwargs: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Propose a change to a protocol"""
        try:
            # Higher reward for taking action and proposing changes
            reward = 1.0  # Increased reward to encourage proposal actions
            
            # Reset analysis count when taking meaningful action
            self._analysis_count = 0
            info = {
                "action": "propose_protocol_change",
                "proposal_generated": True,
                "success": True,  # Mark as successful for proposal collection
                "kwargs": kwargs
            }
            return reward, info
        except Exception as e:
            return -0.2, {"error": str(e), "action": "propose_protocol_change"}

    def _action_evaluate_repository(self) -> Tuple[float, Dict[str, Any]]:
        """Evaluate the overall repository state"""
        try:
            observation = self._get_observation()
            # Reward based on repository health
            reward = 0.3 if observation.get('protocols_count', 0) > 30 else 0.1
            info = {
                "action": "evaluate_repository",
                "repository_health": "good" if reward > 0.2 else "fair",
                "success": reward > 0.2  # Success if repository is healthy
            }
            return reward, info
        except Exception as e:
            return -0.1, {"error": str(e), "action": "evaluate_repository"}

    def _action_read_file(self, file_path: str) -> Tuple[float, Dict[str, Any]]:
        """Read a file from the repository (exploration action)"""
        if not file_path:
            return -0.1, {"error": "No file path provided"}
        
        full_path = self.repo_path / file_path
        
        if not full_path.exists():
            return -0.2, {"error": f"File does not exist: {file_path}"}
        
        if not self._is_allowed_path(file_path):
            return -0.5, {"error": f"Access denied to: {file_path}"}
        
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            return 0.1, {
                "success": True,
                "file_path": file_path,
                "content_length": len(content),
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            }
        except Exception as e:
            return -0.3, {"error": f"Failed to read file: {str(e)}"}
    
    def _is_allowed_path(self, file_path: str) -> bool:
        """Check if The Gardener is allowed to access this path"""
        for allowed_category, allowed_path in self.allowed_paths.items():
            if file_path.startswith(allowed_path) or file_path == allowed_path:
                return True
        return False
    
    def _action_propose_protocol_refinement(self, kwargs: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Propose a refinement to an existing protocol"""
        protocol_path = kwargs.get('protocol_path', '')
        proposed_changes = kwargs.get('proposed_changes', '')
        rationale = kwargs.get('rationale', '')
        
        if not all([protocol_path, proposed_changes, rationale]):
            return -0.3, {"error": "Missing required parameters for protocol refinement"}
        
        if not self._is_allowed_path(protocol_path):
            return -0.5, {"error": f"Cannot modify: {protocol_path}"}
        
        # Read current protocol content
        full_path = self.repo_path / protocol_path
        if not full_path.exists():
            return -0.2, {"error": f"Protocol file does not exist: {protocol_path}"}
        
        with open(full_path, 'r') as f:
            current_content = f.read()
        
        # Create proposed change
        change = ProposedChange(
            file_path=protocol_path,
            old_content=current_content,
            new_content=proposed_changes,
            rationale=rationale,
            protocol_reference="Protocol 37 - The Move 37 Protocol",
            confidence=kwargs.get('confidence', 0.7)
        )
        
        self.current_state.proposed_changes.append(change)
        self.current_state.total_changes_proposed += 1
        
        # Base reward for valid proposal
        reward = 0.5
        
        # Bonus for referencing other protocols
        if "Protocol" in rationale:
            reward += 0.1
        
        # Bonus for coherent reasoning
        if len(rationale) > 100:
            reward += 0.1
        
        return reward, {
            "success": True,
            "change_hash": change.change_hash,
            "proposal_count": len(self.current_state.proposed_changes)
        }
    
    def _action_propose_chronicle_entry(self, kwargs: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Propose a new Chronicle entry"""
        entry_title = kwargs.get('entry_title', '')
        entry_content = kwargs.get('entry_content', '')
        entry_status = kwargs.get('entry_status', 'PROPOSED')
        
        if not all([entry_title, entry_content]):
            return -0.3, {"error": "Missing title or content for chronicle entry"}
        
        # Get next entry number
        next_entry_num = self._get_next_chronicle_entry_number()
        
        # Format the entry according to Chronicle standards
        formatted_entry = self._format_chronicle_entry(
            next_entry_num, entry_title, entry_content, entry_status
        )
        
        # Read current chronicle
        chronicle_path = "Living_Chronicle.md"
        with open(self.repo_path / chronicle_path, 'r') as f:
            current_chronicle = f.read()
        
        # Propose adding the entry
        new_chronicle = current_chronicle + "\n" + formatted_entry
        
        change = ProposedChange(
            file_path=chronicle_path,
            old_content=current_chronicle,
            new_content=new_chronicle,
            rationale=f"Proposed Chronicle Entry: {entry_title}",
            protocol_reference="Living Chronicle Standards",
            confidence=kwargs.get('confidence', 0.8)
        )
        
        self.current_state.proposed_changes.append(change)
        self.current_state.total_changes_proposed += 1
        
        return 0.6, {
            "success": True,
            "entry_number": next_entry_num,
            "change_hash": change.change_hash
        }
    
    def _get_next_chronicle_entry_number(self) -> int:
        """Determine the next Chronicle entry number"""
        chronicle_path = self.repo_path / "Living_Chronicle.md"
        with open(chronicle_path, 'r') as f:
            content = f.read()
        
        # Find the highest entry number
        import re
        entry_pattern = r'### \*\*Entry (\d+):'
        matches = re.findall(entry_pattern, content)
        
        if matches:
            return max(int(match) for match in matches) + 1
        else:
            return 1
    
    def _format_chronicle_entry(self, entry_num: int, title: str, content: str, status: str) -> str:
        """Format a Chronicle entry according to standards"""
        from datetime import datetime
        
        formatted_entry = f"""
### **Entry {entry_num:03d}: {title}**
**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Origin:** The Gardener - Autonomous Cognitive Enhancement
**Status:** **{status}**

{content}

---
"""
        return formatted_entry
    
    def _action_propose_documentation_improvement(self, kwargs: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Propose improvements to documentation"""
        # Implementation for documentation improvements
        file_path = kwargs.get('file_path', '')
        improvement_type = kwargs.get('improvement_type', 'clarity')
        proposed_text = kwargs.get('proposed_text', '')
        
        if not all([file_path, proposed_text]):
            return -0.2, {"error": "Missing parameters for documentation improvement"}
        
        # Basic implementation - can be expanded
        return 0.3, {"success": True, "improvement_type": improvement_type}
    
    def _action_analyze_doctrinal_coherence(self, kwargs: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Analyze coherence between protocols"""
        # Implementation for doctrinal analysis
        protocols_to_analyze = kwargs.get('protocols', [])
        
        if not protocols_to_analyze:
            # Analyze all protocols by default
            protocols_dir = self.repo_path / "01_PROTOCOLS"
            protocols_to_analyze = list(protocols_dir.glob("*.md"))
        
        # Basic coherence analysis - can be expanded with NLP
        coherence_score = 0.75  # Placeholder
        
        return 0.4, {
            "success": True,
            "coherence_score": coherence_score,
            "protocols_analyzed": len(protocols_to_analyze)
        }
    
    def _action_submit_for_jury_review(self) -> Tuple[float, Dict[str, Any]]:
        """Submit current proposed changes for Hybrid Jury review"""
        if not self.current_state.proposed_changes:
            return -0.2, {"error": "No changes to submit for review"}
        
        # Create a simulated PR for jury review
        pr_branch = f"gardener-proposal-{self.current_state.episode_number}"
        
        try:
            # Apply all proposed changes to files
            file_paths = []
            for change in self.current_state.proposed_changes:
                file_path = self.repo_path / change.file_path
                with open(file_path, 'w') as f:
                    f.write(change.new_content)
                file_paths.append(change.file_path)
            
            # Stage and commit the changes
            self.git.add_files(file_paths)
            
            commit_message = f"The Gardener's Proposal - Episode {self.current_state.episode_number}"
            commit_message += f"\n\nProposed changes:\n"
            for change in self.current_state.proposed_changes:
                commit_message += f"- {change.file_path}: {change.rationale}\n"
            
            commit_hash = self.git.commit(commit_message)
            
            # Simulate jury verdict (placeholder - real implementation would integrate with jury system)
            jury_verdict = self._simulate_jury_verdict()
            self.current_state.last_jury_verdict = jury_verdict
            
            if jury_verdict == "ACCEPT":
                self.current_state.successful_merges += 1
                reward = 2.0
            elif jury_verdict == "ACCEPT_WITH_REFINEMENTS":
                reward = 1.0
            else:  # REJECT
                reward = -1.0
            
            return reward, {
                "success": True,
                "jury_verdict": jury_verdict,
                "changes_submitted": len(self.current_state.proposed_changes),
                "commit_hash": commit_hash
            }
            
        except Exception as e:
            return -0.5, {"error": f"Failed to submit for review: {str(e)}"}

    def _action_propose_protocol_from_template(self, kwargs: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Propose a protocol based on a predefined template - The Refined Architect's Forge
        
        The Artisan and Power Saw Doctrine:
        - Artisans (Council) create templates and identify architectural needs
        - Power Saw (Gardener) excels at pattern recognition, gap identification, and template application
        
        High reward (+100) for excellent gap identification and template application.
        Origin: The Refined Architect's Forge (Council Wisdom Synthesis)
        """
        template_type = kwargs.get('template_type', '')
        gap_identified = kwargs.get('gap_identified', '')
        template_data = kwargs.get('template_data', {})
        confidence = kwargs.get('confidence', 0.7)
        
        if not all([template_type, gap_identified]):
            return -5.0, {"error": "Missing required parameters: template_type and gap_identified"}
        
        # Validate template type exists
        available_templates = self._get_available_templates()
        if template_type not in available_templates:
            return -3.0, {
                "error": f"Unknown template type: {template_type}",
                "available_templates": list(available_templates.keys())
            }
        
        # Validate gap identification quality
        gap_quality_score = self._assess_gap_identification(gap_identified)
        if gap_quality_score < 0.3:
            return -2.0, {"error": "Gap identification insufficient - must clearly identify specific missing capability"}
        
        # Get template and validate data completeness
        template = available_templates[template_type]
        required_fields = template.get('required_fields', [])
        missing_fields = [field for field in required_fields if field not in template_data]
        
        if missing_fields:
            return -1.0, {
                "error": f"Missing required template fields: {missing_fields}",
                "required_fields": required_fields
            }
        
        # Generate protocol from template
        try:
            protocol_content = self._apply_template(template, template_data)
            next_protocol_num = self._get_next_protocol_number()
            
            # Create protocol filename based on template data
            protocol_title = template_data.get('protocol_name', f"{template_type}_Protocol_{next_protocol_num}")
            protocol_filename = f"{next_protocol_num:02d}_{protocol_title.replace(' ', '_')}.md"
            protocol_path = f"01_PROTOCOLS/{protocol_filename}"
            
            # Assess template application quality
            application_score = self._assess_template_application(template_data, template)
            relevance_score = self._assess_gap_relevance(gap_identified, template_type)
            
            # Calculate total quality score
            total_quality = (gap_quality_score + application_score + relevance_score) / 3.0
            
            # Create proposed change
            change = ProposedChange(
                file_path=protocol_path,
                old_content="",  # New file
                new_content=protocol_content,
                rationale=f"Template-based Protocol Proposal: {protocol_title}\\n\\nGap Identified: {gap_identified}\\n\\nTemplate Applied: {template_type}",
                protocol_reference="The Refined Architect's Forge - Template-Based Protocol Generation",
                confidence=min(0.95, total_quality)
            )
            
            self.current_state.proposed_changes.append(change)
            self.current_state.total_changes_proposed += 1
            
            # Calculate reward based on quality (same tier system, more achievable)
            if total_quality >= 0.8:
                # ARCHITECT TIER: Excellent gap identification + perfect template application
                reward = 100.0
            elif total_quality >= 0.6:
                # JOURNEYMAN TIER: Good gap identification + solid template application
                reward = 50.0
            elif total_quality >= 0.4:
                # APPRENTICE TIER: Decent gap identification + adequate template application
                reward = 20.0
            else:
                # NOVICE TIER: Basic gap identification + template application needs work
                reward = 5.0
            
            return reward, {
                "success": True,
                "action": "propose_protocol_from_template",
                "protocol_number": next_protocol_num,
                "protocol_title": protocol_title,
                "protocol_path": protocol_path,
                "template_type": template_type,
                "quality_scores": {
                    "gap_identification": gap_quality_score,
                    "template_application": application_score,
                    "relevance": relevance_score,
                    "total_quality": total_quality
                },
                "reward_tier": (
                    "ARCHITECT" if total_quality >= 0.8 else
                    "JOURNEYMAN" if total_quality >= 0.6 else
                    "APPRENTICE" if total_quality >= 0.4 else
                    "NOVICE"
                ),
                "change_hash": change.change_hash,
                "proposal_count": len(self.current_state.proposed_changes)
            }
            
        except Exception as e:
            return -1.0, {"error": f"Template application failed: {str(e)}"}
    
    def _get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get available protocol templates for The Gardener to use"""
        return {
            "security_protocol": {
                "name": "Security Protocol Template",
                "description": "For protocols addressing security, access control, and data protection",
                "required_fields": ["security_objective", "threat_model", "controls", "verification_method"],
                "template": """## 📜 I. Security Objective

{security_objective}

## ⚡ II. Threat Model

{threat_model}

## ⚙️ III. Security Controls

{controls}

## 🔍 IV. Verification & Compliance

{verification_method}

## 📋 V. Implementation

This security protocol requires Council review and integration with existing security framework."""
            },
            
            "workflow_protocol": {
                "name": "Workflow Protocol Template", 
                "description": "For protocols defining operational procedures and workflows",
                "required_fields": ["workflow_purpose", "trigger_conditions", "process_steps", "success_criteria"],
                "template": """## 📜 I. Workflow Purpose

{workflow_purpose}

## ⚡ II. Trigger Conditions

{trigger_conditions}

## ⚙️ III. Process Steps

{process_steps}

## ✅ IV. Success Criteria

{success_criteria}

## 📋 V. Implementation

This workflow protocol requires Council approval and integration with operational procedures."""
            },
            
            "governance_protocol": {
                "name": "Governance Protocol Template",
                "description": "For protocols addressing decision-making, oversight, and accountability",
                "required_fields": ["governance_scope", "decision_authority", "oversight_mechanism", "accountability_measures"],
                "template": """## 📜 I. Governance Scope

{governance_scope}

## ⚡ II. Decision Authority

{decision_authority}

## ⚙️ III. Oversight Mechanism

{oversight_mechanism}

## 🔍 IV. Accountability Measures

{accountability_measures}

## 📋 V. Implementation

This governance protocol requires Council ratification and integration with existing governance framework."""
            },
            
            "integration_protocol": {
                "name": "Integration Protocol Template",
                "description": "For protocols addressing system integration and interoperability",
                "required_fields": ["integration_purpose", "system_components", "interface_specifications", "testing_requirements"],
                "template": """## 📜 I. Integration Purpose

{integration_purpose}

## ⚡ II. System Components

{system_components}

## ⚙️ III. Interface Specifications

{interface_specifications}

## 🔍 IV. Testing & Validation

{testing_requirements}

## 📋 V. Implementation

This integration protocol requires technical review and testing before deployment."""
            }
        }
    
    def _apply_template(self, template: Dict[str, Any], template_data: Dict[str, str]) -> str:
        """Apply template data to generate protocol content"""
        template_content = template["template"]
        
        # Replace placeholders with actual data
        for field, value in template_data.items():
            placeholder = "{" + field + "}"
            template_content = template_content.replace(placeholder, value)
        
        # Generate full protocol with proper header
        protocol_name = template_data.get('protocol_name', 'Generated_Protocol')
        protocol_class = template_data.get('protocol_class', template['name'].split()[0])
        
        header = f"""# {self._get_next_protocol_number():02d}_{protocol_name.replace(' ', '_')}.md

## {protocol_name} - v1.0

**Status:** Proposed | **Protocol Class:** {protocol_class} | **Version:** 1.0  
**Origin:** The Refined Architect's Forge - Template-Based Generation  
**Template:** {template['name']}  

---
"""
        
        footer = """
---

**Glass Box Transparency:** This protocol was generated by The Gardener using template-based drafting, demonstrating the Power Saw approach to rapid, high-quality protocol creation under Artisan guidance.
"""
        
        return header + template_content + footer
    
    def _assess_gap_identification(self, gap_description: str) -> float:
        """Assess the quality of gap identification"""
        score = 0.0
        gap_lower = gap_description.lower()
        
        # Check for specific gap identification
        if any(word in gap_lower for word in ["gap", "missing", "lacking", "absent", "no protocol", "need"]):
            score += 0.3
        
        # Check for consequence analysis
        if any(word in gap_lower for word in ["without", "problem", "risk", "issue", "challenge"]):
            score += 0.3
        
        # Check for specificity
        if len(gap_description) > 100:
            score += 0.2
        
        # Check for domain knowledge
        if any(word in gap_lower for word in ["security", "workflow", "governance", "integration", "protocol"]):
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_template_application(self, template_data: Dict[str, str], template: Dict[str, Any]) -> float:
        """Assess how well the template was applied"""
        score = 0.0
        
        # Check completeness - all required fields present
        required_fields = template.get('required_fields', [])
        if all(field in template_data for field in required_fields):
            score += 0.4
        
        # Check content quality - non-trivial responses
        meaningful_responses = 0
        for field in required_fields:
            if field in template_data and len(template_data[field]) > 50:
                meaningful_responses += 1
        
        if meaningful_responses >= len(required_fields) * 0.8:
            score += 0.3
        
        # Check for protocol naming
        if 'protocol_name' in template_data and len(template_data['protocol_name']) > 5:
            score += 0.2
        
        # Check for proper classification
        if 'protocol_class' in template_data:
            score += 0.1
        
        return min(score, 1.0)
    
    def _assess_gap_relevance(self, gap_description: str, template_type: str) -> float:
        """Assess how well the identified gap matches the chosen template"""
        gap_lower = gap_description.lower()
        
        # Template-specific keyword matching
        template_keywords = {
            "security_protocol": ["security", "access", "protection", "authentication", "authorization", "encryption"],
            "workflow_protocol": ["process", "procedure", "workflow", "steps", "operation", "task"],
            "governance_protocol": ["decision", "oversight", "accountability", "authority", "governance", "control"],
            "integration_protocol": ["integration", "interface", "connection", "compatibility", "interop", "system"]
        }
        
        keywords = template_keywords.get(template_type, [])
        matches = sum(1 for keyword in keywords if keyword in gap_lower)
        
        return min(matches / len(keywords), 1.0) if keywords else 0.5
    
    def _get_next_protocol_number(self) -> int:
        """Determine the next available protocol number"""
        protocols_dir = self.repo_path / "01_PROTOCOLS"
        existing_protocols = []
        
        if protocols_dir.exists():
            for file_path in protocols_dir.glob("*.md"):
                filename = file_path.name
                if filename[:2].isdigit():
                    try:
                        protocol_num = int(filename[:2])
                        existing_protocols.append(protocol_num)
                    except ValueError:
                        continue
        
        return max(existing_protocols, default=0) + 1
    
    def _analyze_protocol_redundancy(self, content: str, gap_analysis: str) -> float:
        """Analyze if the proposed protocol is redundant with existing ones"""
        # Simplified implementation - in practice would use more sophisticated NLP
        existing_protocols_dir = self.repo_path / "01_PROTOCOLS"
        
        if not existing_protocols_dir.exists():
            return 0.0
        
        content_lower = content.lower()
        gap_lower = gap_analysis.lower()
        combined_text = f"{content_lower} {gap_lower}"
        
        max_similarity = 0.0
        
        for protocol_file in existing_protocols_dir.glob("*.md"):
            try:
                with open(protocol_file, 'r') as f:
                    existing_content = f.read().lower()
                
                # Simple keyword overlap analysis
                existing_words = set(existing_content.split())
                proposed_words = set(combined_text.split())
                
                if len(proposed_words) == 0:
                    continue
                
                overlap = len(existing_words.intersection(proposed_words))
                similarity = overlap / len(proposed_words)
                max_similarity = max(max_similarity, similarity)
                
            except Exception:
                continue
        
        return min(max_similarity, 1.0)
    
    def _assess_protocol_coherence(self, content: str, rationale: str) -> float:
        """Assess the internal coherence and quality of the protocol"""
        score = 0.0
        
        # Check for structured content
        if "preamble" in content.lower() or "## " in content:
            score += 0.2
        
        # Check for doctrinal references
        if "protocol" in content.lower():
            score += 0.2
        
        # Check for principle-based reasoning
        if any(word in content.lower() for word in ["principle", "doctrine", "wisdom", "governance"]):
            score += 0.2
        
        # Check for implementation details
        if any(word in content.lower() for word in ["implementation", "procedure", "process", "step"]):
            score += 0.2
        
        # Check rationale quality
        if len(rationale) > 200 and "because" in rationale.lower():
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_gap_necessity(self, gap_analysis: str) -> float:
        """Assess how well the gap analysis demonstrates necessity"""
        score = 0.0
        gap_lower = gap_analysis.lower()
        
        # Check for specific gap identification
        if any(word in gap_lower for word in ["gap", "missing", "lacking", "absent", "need"]):
            score += 0.3
        
        # Check for consequence analysis
        if any(word in gap_lower for word in ["without", "problem", "issue", "challenge", "risk"]):
            score += 0.3
        
        # Check for solution connection
        if any(word in gap_lower for word in ["therefore", "thus", "hence", "solution", "address"]):
            score += 0.2
        
        # Check for specificity
        if len(gap_analysis) > 150:
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_protocol_markdown(self, protocol_num: int, title: str, content: str, 
                                  protocol_class: str, gap_analysis: str, rationale: str) -> str:
        """Generate properly formatted protocol markdown"""
        formatted_title = title.replace('_', ' ').title()
        
        return f"""# {protocol_num:02d}_{title.replace(' ', '_')}.md

## {formatted_title} - v1.0

**Status:** Proposed | **Protocol Class:** {protocol_class} | **Version:** 1.0  
**Origin:** Operation: The Architect's Forge - Autonomous Protocol Generation  
**Proposer:** The Gardener (Autonomous Agent)  

---

## 📜 I. Gap Analysis

{gap_analysis}

---

## ⚙️ II. The Protocol

{content}

---

## 📋 III. Implementation

This protocol requires Council review and formal ratification before implementation.

**Rationale:** {rationale}

**Integration Points:** This protocol should be integrated with existing Sanctuary doctrine after Council approval.

---

**Glass Box Transparency:** This protocol was autonomously generated by The Gardener neural network as part of Operation: The Architect's Forge, demonstrating the system's capacity for architectural reasoning and doctrinal evolution.
"""
    
    def _simulate_jury_verdict(self) -> str:
        """Simulate Hybrid Jury verdict (placeholder for real jury integration)"""
        # This is a simplified simulation - real implementation would
        # integrate with the actual Hybrid Jury system
        
        import random
        
        # Basic heuristics for verdict
        total_changes = len(self.current_state.proposed_changes)
        
        if total_changes == 0:
            return "REJECT"
        
        # Simple scoring based on change quality indicators
        score = 0
        for change in self.current_state.proposed_changes:
            if len(change.rationale) > 50:
                score += 1
            if "Protocol" in change.rationale:
                score += 1
            if change.confidence > 0.7:
                score += 1
        
        avg_score = score / (total_changes * 3)  # Normalize
        
        if avg_score > 0.7:
            return "ACCEPT"
        elif avg_score > 0.4:
            return "ACCEPT_WITH_REFINEMENTS"
        else:
            return "REJECT"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get environment metrics for monitoring"""
        return {
            "episode_number": self.current_state.episode_number,
            "total_changes_proposed": self.current_state.total_changes_proposed,
            "successful_merges": self.current_state.successful_merges,
            "success_rate": self.current_state.successful_merges / max(1, self.current_state.episode_number),
            "current_pending_changes": len(self.current_state.proposed_changes),
            "last_jury_verdict": self.current_state.last_jury_verdict
        }


# Import pandas for timestamp logging
try:
    import pandas as pd
except ImportError:
    # Fallback if pandas not available
    class pd:
        @staticmethod
        class Timestamp:
            @staticmethod
            def now():
                from datetime import datetime
                return datetime.now()
