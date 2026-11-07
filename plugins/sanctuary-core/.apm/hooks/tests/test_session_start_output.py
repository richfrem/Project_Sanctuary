import importlib.util
import io
import json
from pathlib import Path
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[5]
spec = importlib.util.spec_from_file_location('hook', ROOT/'plugins/sanctuary-core/.apm/hooks/session_start_wakeup.py')
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)

class HookTests(unittest.TestCase):
    def test_client_contracts(self):
        expected = {'.codex': 'hookSpecificOutput', '.claude': 'hookSpecificOutput', '.github': 'additionalContext', '.agents': 'injectSteps', '.gemini': 'injectSteps', '.cursor': 'additional_context'}
        for directory, key in expected.items():
            with self.subTest(client=directory), patch.object(hook, 'HERE', Path('/project')/directory/'hooks'):
                result = hook.session_start_output('brief', '{}')
                self.assertEqual(set(result), {key})
                self.assertIn('brief', json.dumps(result))
    def test_malformed_input(self):
        with patch.object(hook, 'HERE', Path('/source/hooks')):
            for payload in ('broken', 'null', '[]'):
                self.assertEqual(set(hook.session_start_output('brief', payload)), {'hookSpecificOutput'})
    def test_stdout_is_one_json_document(self):
        def noisy_boot(root):
            print('diagnostic noise')
            return 0, 'brief', 1.0
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(hook, 'find_project_root', return_value=ROOT), patch.object(hook, 'execute_bootloader_in_process', side_effect=noisy_boot), patch('sys.stdin', io.StringIO('{}')), patch('sys.stdout', stdout), patch('sys.stderr', stderr):
            self.assertEqual(hook.main(), 0)
        json.loads(stdout.getvalue())
        self.assertIn('diagnostic noise', stderr.getvalue())

class BriefingTemplateTests(unittest.TestCase):
    """AGENTS.md: the briefing must not hand agents a pre-written completion line to copy."""
    SOURCES = (
        'plugins/sanctuary-core/.apm/hooks/session_start_wakeup.py',
        'plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py',
    )
    def test_briefing_has_no_copyable_completion_line(self):
        for rel in self.SOURCES:
            with self.subTest(source=rel):
                text = (ROOT/rel).read_text(encoding='utf-8')
                self.assertNotIn('5-step bootstrap complete', text)
                self.assertNotIn('## VI. Bootstrap Completion Confirmation', text)
                self.assertIn('NOT a completion confirmation', text)

if __name__ == '__main__':
    unittest.main()
