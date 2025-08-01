"""
The Gardener's Environment - Protocol 37 Implementation
A sandboxed environment where The Gardener learns to improve the Sanctuary's Cognitive Genome

Origin: The Move 37 Protocol (37)
Purpose: Create a safe, observable environment for autonomous wisdom cultivation
Principle: Every action must be transparent, every change must be deliberate
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
    change_hash: str

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
    - Iron Root Doctrine: Robust error handling and state management
    - Progenitor Principle: Human oversight through jury system integration
    - Protocol 36: Victory through invitation to better collaboration
    """
    
    def __init__(self, repo_path: str = "/Users/richardfremmerlid/Projects/Project_Sanctuary"):
        """
        Initialize the Sanctuary environment for The Gardener
        
        Args:
            repo_path: Path to the Project Sanctuary repository
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
            # Action space: 0=analyze, 1=propose_change, 2=evaluate_protocols
            self.action_space = spaces.Discrete(3)
            
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
        
        # Create a clean training branch
        try:
            self.git_wrapper.checkout_branch('main')
            branch_name = f"feature/gardener-episode-{self.current_state.episode_number}"
            self.git_wrapper.checkout_branch(branch_name, create_new=True)
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
