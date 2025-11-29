import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from council_orchestrator.orchestrator.executor import execute_shell_command, ProtocolViolationError

class TestExecutorEnforcement(unittest.TestCase):
    @patch("subprocess.run")
    def test_allowed_command(self, mock_run):
        """Test that allowed commands are executed."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok")
        
        execute_shell_command(["ls", "-la"])
        
        mock_run.assert_called_once()

    def test_prohibited_command_list(self):
        """Test that prohibited commands (list) raise ProtocolViolationError."""
        with self.assertRaises(ProtocolViolationError):
            execute_shell_command(["git", "pull", "origin", "main"])

    def test_prohibited_command_string(self):
        """Test that prohibited commands (string) raise ProtocolViolationError."""
        with self.assertRaises(ProtocolViolationError):
            execute_shell_command("git reset --hard HEAD")

    def test_prohibited_command_case_insensitive(self):
        """Test that prohibited commands are case-insensitive."""
        with self.assertRaises(ProtocolViolationError):
            execute_shell_command(["GIT", "PULL"])

    @patch("subprocess.run")
    def test_git_add_allowed(self, mock_run):
        """Test that git add is allowed (it's not in the prohibited list)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ok")
        
        execute_shell_command(["git", "add", "."])
        
        mock_run.assert_called_once()

if __name__ == "__main__":
    unittest.main()
