#!/usr/bin/env python3
"""
security_scan.py - Shift Left Security Scanner for Project Sanctuary

This script runs dependency vulnerability scans locally before pushing to GitHub.
It checks for known vulnerabilities in Python dependencies defined in the
project's requirements.txt file.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class SecurityScanner:
    """Main security scanner class."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.requirements_file = project_root / "requirements.txt"
        self.report_file = project_root / "security_report.md"

    def run_safety_scan(self) -> Tuple[List[Dict], bool]:
        """Runs the 'safety' vulnerability scan."""
        print("🔍 Running Safety vulnerability scan on requirements.txt...")

        if not self.requirements_file.exists():
            print(f"❌ ERROR: requirements.txt not found at '{self.requirements_file}'!")
            return [], False

        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "safety"], check=True, capture_output=True)
            scan_cmd = [sys.executable, "-m", "safety", "check", "--file", str(self.requirements_file), "--json"]
            result = subprocess.run(scan_cmd, capture_output=True, text=True)

            if result.returncode in [0, 1]:
                vulnerabilities = json.loads(result.stdout) if result.stdout else []
                if vulnerabilities:
                    print(f"🚨 Found {len(vulnerabilities)} vulnerabilities!")
                else:
                    print("✅ No vulnerabilities found.")
                return vulnerabilities, True
            else:
                print("❌ Safety scan command failed unexpectedly.")
                print("STDERR:", result.stderr)
                return [], False
        except (subprocess.CalledProcessError, json.JSONDecodeError, Exception) as e:
            print(f"❌ An error occurred during the safety scan: {e}")
            return [], False

    def generate_report(self, vulnerabilities: List[Dict]) -> str:
        """Generates a security report in Markdown format."""
        report = [
            "# 🔒 Project Sanctuary - Security Scan Report",
            f"**Scan Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**File Scanned:** {self.requirements_file.name}",
            "\n---",
        ]

        if not vulnerabilities:
            report.append("## 📊 Summary\n\n**✅ No vulnerabilities found.**")
            return "\n".join(report)

        summary = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for vuln in vulnerabilities:
            # The safety JSON format is a list of lists/tuples
            # [package, affected, installed, description, cve, severity]
            severity = str(vuln[5]).upper() if len(vuln) > 5 and vuln[5] else "UNKNOWN"
            if severity in summary:
                summary[severity] += 1

        report.append("## 📊 Summary")
        report.append(f"- **Total Vulnerabilities Found:** {len(vulnerabilities)}")
        for severity, count in summary.items():
            if count > 0:
                report.append(f"- **{severity}:** {count}")

        report.append("\n## 🚨 Vulnerability Details")
        # Sort by severity - assuming severity is the 6th element
        for vuln in sorted(vulnerabilities, key=lambda x: str(x[5] or ''), reverse=True):
            package, affected, installed, description, cve, severity = vuln
            report.append(f"\n### {str(severity).upper()}: {package} (CVE: {cve or 'N/A'})")
            report.append(f"- **Installed Version:** {installed}")
            report.append(f"- **Affected Versions:** {affected}")
            report.append(f"- **Description:** {description}")

        return "\n".join(report)

    def run_scan(self, ci_mode: bool = False) -> int:
        """Runs the scan, generates a report, and returns an exit code."""
        print("🚀 Starting Shift-Left Security Scan for Project Sanctuary")
        print("=" * 60)

        vulnerabilities, scan_success = self.run_safety_scan()

        if not scan_success:
            print("❌ Scan failed. Aborting.")
            return 1

        report_content = self.generate_report(vulnerabilities)
        
        try:
            with open(self.report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"\n📝 Full report saved to: {self.report_file}")
        except Exception as e:
            print(f"⚠️ Could not write report file: {e}")

        has_critical_high = any(str(v[5]).upper() in ["CRITICAL", "HIGH"] for v in vulnerabilities if len(v) > 5)

        if not vulnerabilities:
            print("\n✅ Security scan passed.")
            return 0
        
        print("\n" + report_content)

        if ci_mode and has_critical_high:
            print("\n❌ CI MODE: Critical/High vulnerabilities found. Failing build.")
            return 1
        else:
            print("\n⚠️ Vulnerabilities detected. Please review the report.")
            return 1 if ci_mode else 0


def main():
    parser = argparse.ArgumentParser(description="Shift-Left Security Scanner for Project Sanctuary.")
    parser.add_argument("--ci", action="store_true", help="CI mode - exit with error code on high/critical vulnerabilities.")
    args = parser.parse_args()

    # The script is in tools/, so the project root is its parent directory.
    project_root = Path(__file__).resolve().parent.parent
    scanner = SecurityScanner(project_root)
    exit_code = scanner.run_scan(ci_mode=args.ci)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()