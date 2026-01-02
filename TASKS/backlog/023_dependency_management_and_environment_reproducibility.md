# Task 023: Dependency Management & Environment Reproducibility

## Metadata
- **Status**: backlog
- **Priority**: high
- **Complexity**: medium
- **Category**: infrastructure
- **Estimated Effort**: 6-8 hours
- **Dependencies**: None
- **Assigned To**: Unassigned
- **Created**: 2025-11-21
- **Target Completion**: TBD

## Context

Project Sanctuary has achieved unified dependency management through `setup_cuda_env.py`, but several areas need enhancement for true reproducibility and maintainability:

**Current State:**
- ✅ Unified `requirements.txt` with 180+ pinned dependencies
- ✅ Comprehensive `DEPENDENCY_MANIFEST.md` documentation
- ✅ Automated setup script with surgical CUDA installations
- ❌ No dependency vulnerability scanning
- ❌ No automated dependency updates
- ❌ No dependency license compliance checking
- ❌ No dependency graph visualization
- ❌ No alternative dependency sources for offline/air-gapped environments

**Strategic Alignment:**
- **Protocol 89**: The Clean Forge - Dependencies must be clean and verifiable
- **Protocol 101**: The Unbreakable Commit - Reproducible builds are mandatory
- **Protocol 115**: The Tactical Mandate - Dependency management is tactical

## Objective

Enhance dependency management and environment reproducibility with focus on:
1. Automated dependency security scanning
2. Dependency update automation with testing
3. License compliance verification
4. Offline/air-gapped environment support
5. Dependency graph visualization and analysis

## Acceptance Criteria

### 1. Security Scanning
- [ ] Integrate `pip-audit` for vulnerability scanning
- [ ] Create `tools/deps/scan_vulnerabilities.py` script
- [ ] Add vulnerability scanning to CI/CD pipeline
- [ ] Create vulnerability report template
- [ ] Establish vulnerability remediation workflow
- [ ] Add security scanning badge to README

### 2. Automated Dependency Updates
- [ ] Set up Dependabot or Renovate for automated PRs
- [ ] Create `tools/deps/update_dependencies.py` script
- [ ] Implement automated testing for dependency updates
- [ ] Create dependency update policy document
- [ ] Add dependency update schedule (weekly/monthly)
- [ ] Implement rollback mechanism for failed updates

### 3. License Compliance
- [ ] Create `tools/deps/check_licenses.py` script
- [ ] Generate `DEPENDENCY_LICENSES.md` report
- [ ] Identify incompatible licenses
- [ ] Create license compliance policy
- [ ] Add license scanning to CI/CD
- [ ] Document license compatibility matrix

### 4. Offline Environment Support
- [ ] Create `tools/deps/create_offline_bundle.py` script
- [ ] Generate wheel bundle for all dependencies
- [ ] Create offline installation guide
- [ ] Test offline installation in isolated environment
- [ ] Document air-gapped deployment process
- [ ] Create dependency mirror setup guide

### 5. Dependency Analysis
- [ ] Create `tools/deps/analyze_dependencies.py` script
- [ ] Generate dependency graph visualization
- [ ] Identify circular dependencies
- [ ] Detect unused dependencies
- [ ] Calculate dependency size metrics
- [ ] Create dependency health report

### 6. Environment Validation
- [ ] Create `tools/deps/validate_environment.py` script
- [ ] Verify all dependencies are correctly installed
- [ ] Check for version conflicts
- [ ] Validate CUDA/GPU dependencies
- [ ] Test import of all critical modules
- [ ] Generate environment health report

### 7. Documentation Updates
- [ ] Update `docs/operations/DEPENDENCY_MANIFEST.md` with security info
- [ ] Create `docs/DEPENDENCY_MANAGEMENT.md` guide
- [ ] Document dependency update workflow
- [ ] Create troubleshooting guide for dependency issues
- [ ] Add dependency FAQs

## Technical Approach

### Phase 1: Security Scanning (2 hours)
```python
# tools/deps/scan_vulnerabilities.py
"""
Automated dependency vulnerability scanning.

Uses pip-audit to scan for known vulnerabilities in dependencies.
Generates report and optionally fails on high-severity issues.
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict

def scan_vulnerabilities(
    requirements_file: str = "requirements.txt",
    fail_on_severity: str = "high"
) -> Dict:
    """
    Scan dependencies for vulnerabilities.
    
    Args:
        requirements_file: Path to requirements file
        fail_on_severity: Fail if vulnerabilities of this severity found
    
    Returns:
        Dictionary with scan results
    """
    result = subprocess.run(
        ["pip-audit", "-r", requirements_file, "--format", "json"],
        capture_output=True,
        text=True
    )
    
    vulnerabilities = json.loads(result.stdout)
    
    # Generate report
    report = {
        "total_packages": len(vulnerabilities.get("dependencies", [])),
        "vulnerable_packages": 0,
        "high_severity": 0,
        "medium_severity": 0,
        "low_severity": 0,
        "details": []
    }
    
    for vuln in vulnerabilities.get("vulnerabilities", []):
        report["vulnerable_packages"] += 1
        severity = vuln.get("severity", "unknown").lower()
        
        if severity == "high":
            report["high_severity"] += 1
        elif severity == "medium":
            report["medium_severity"] += 1
        elif severity == "low":
            report["low_severity"] += 1
        
        report["details"].append({
            "package": vuln.get("name"),
            "version": vuln.get("version"),
            "vulnerability": vuln.get("id"),
            "severity": severity,
            "fix_available": vuln.get("fix_versions", [])
        })
    
    # Save report
    report_path = Path("reports/vulnerability_scan.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    
    # Fail if high severity found
    if fail_on_severity == "high" and report["high_severity"] > 0:
        raise RuntimeError(f"Found {report['high_severity']} high-severity vulnerabilities!")
    
    return report

if __name__ == "__main__":
    report = scan_vulnerabilities()
    print(f"Scanned {report['total_packages']} packages")
    print(f"Found {report['vulnerable_packages']} vulnerable packages")
    print(f"High: {report['high_severity']}, Medium: {report['medium_severity']}, Low: {report['low_severity']}")
```

### Phase 2: License Compliance (2 hours)
```python
# tools/deps/check_licenses.py
"""
Automated license compliance checking.

Scans all dependencies for license information and checks
against approved license list.
"""

import subprocess
import json
from typing import List, Dict

# Approved licenses (aligned with CC0/CC BY 4.0)
APPROVED_LICENSES = [
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "ISC",
    "Python Software Foundation License",
    "Apache Software License",
]

INCOMPATIBLE_LICENSES = [
    "GPL-3.0",  # Copyleft - incompatible with proprietary use
    "AGPL-3.0",  # Strong copyleft
]

def check_licenses() -> Dict:
    """
    Check licenses of all dependencies.
    
    Returns:
        Dictionary with license compliance report
    """
    # Use pip-licenses to get license info
    result = subprocess.run(
        ["pip-licenses", "--format=json"],
        capture_output=True,
        text=True
    )
    
    packages = json.loads(result.stdout)
    
    report = {
        "total_packages": len(packages),
        "approved": [],
        "unknown": [],
        "incompatible": [],
        "needs_review": []
    }
    
    for pkg in packages:
        name = pkg.get("Name")
        license_type = pkg.get("License", "Unknown")
        
        if license_type in INCOMPATIBLE_LICENSES:
            report["incompatible"].append({
                "name": name,
                "license": license_type
            })
        elif license_type in APPROVED_LICENSES:
            report["approved"].append({
                "name": name,
                "license": license_type
            })
        elif license_type == "Unknown":
            report["unknown"].append({
                "name": name,
                "license": license_type
            })
        else:
            report["needs_review"].append({
                "name": name,
                "license": license_type
            })
    
    # Generate markdown report
    generate_license_report(report)
    
    # Fail if incompatible licenses found
    if report["incompatible"]:
        raise RuntimeError(f"Found {len(report['incompatible'])} packages with incompatible licenses!")
    
    return report

def generate_license_report(report: Dict):
    """Generate DEPENDENCY_LICENSES.md report."""
    content = f"""# Dependency License Report

Generated: {datetime.datetime.now().isoformat()}

## Summary

- **Total Packages**: {report['total_packages']}
- **Approved Licenses**: {len(report['approved'])}
- **Needs Review**: {len(report['needs_review'])}
- **Unknown Licenses**: {len(report['unknown'])}
- **Incompatible Licenses**: {len(report['incompatible'])}

## Approved Licenses

| Package | License |
|---------|---------|
"""
    
    for pkg in report["approved"]:
        content += f"| {pkg['name']} | {pkg['license']} |\n"
    
    # Add other sections...
    
    Path("DEPENDENCY_LICENSES.md").write_text(content)

if __name__ == "__main__":
    report = check_licenses()
    print(f"License compliance check complete")
    print(f"Approved: {len(report['approved'])}, Needs Review: {len(report['needs_review'])}")
```

### Phase 3: Offline Bundle Creation (2 hours)
```python
# tools/deps/create_offline_bundle.py
"""
Create offline dependency bundle for air-gapped environments.

Downloads all wheels and creates installation script.
"""

import subprocess
from pathlib import Path

def create_offline_bundle(
    requirements_file: str = "requirements.txt",
    output_dir: str = "offline_bundle"
):
    """
    Create offline installation bundle.
    
    Args:
        requirements_file: Path to requirements file
        output_dir: Directory to store bundle
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Download all wheels
    subprocess.run([
        "pip", "download",
        "-r", requirements_file,
        "-d", str(output_path / "wheels"),
        "--platform", "win_amd64",
        "--python-version", "311"
    ])
    
    # Create installation script
    install_script = """#!/bin/bash
# Offline installation script for Project Sanctuary

echo "Installing dependencies from offline bundle..."
pip install --no-index --find-links=wheels -r requirements.txt
echo "Installation complete!"
"""
    
    (output_path / "install.sh").write_text(install_script)
    
    # Create Windows installation script
    install_ps1 = """# Offline installation script for Project Sanctuary (Windows)

Write-Host "Installing dependencies from offline bundle..."
pip install --no-index --find-links=wheels -r requirements.txt
Write-Host "Installation complete!"
"""
    
    (output_path / "install.ps1").write_text(install_ps1)
    
    # Copy requirements.txt
    import shutil
    shutil.copy(requirements_file, output_path / "requirements.txt")
    
    print(f"Offline bundle created in {output_dir}/")
    print(f"Bundle size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    create_offline_bundle()
```

### Phase 4: CI/CD Integration
```yaml
# .github/workflows/dependency-check.yml
name: Dependency Security & Compliance

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  pull_request:
    paths:
      - 'requirements.txt'
      - 'setup_cuda_env.py'

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install scanning tools
        run: |
          pip install pip-audit pip-licenses
      
      - name: Scan for vulnerabilities
        run: |
          python tools/deps/scan_vulnerabilities.py
      
      - name: Check license compliance
        run: |
          python tools/deps/check_licenses.py
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: dependency-reports
          path: reports/
```

## Testing Strategy

### Validation Tests
- [ ] Vulnerability scanner detects known CVEs
- [ ] License checker identifies incompatible licenses
- [ ] Offline bundle installs successfully in isolated environment
- [ ] Environment validator catches missing dependencies
- [ ] Dependency graph generator produces valid output

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| False positive vulnerabilities | Medium | Manual review process, whitelist |
| Dependency update breaks code | High | Automated testing, gradual rollout |
| License compliance overhead | Low | Automated scanning, clear policy |
| Offline bundle size | Low | Compression, selective bundling |

## Success Metrics

- [ ] Zero high-severity vulnerabilities in dependencies
- [ ] 100% license compliance
- [ ] Automated vulnerability scanning in CI/CD
- [ ] Offline bundle tested and documented
- [ ] Dependency graph visualization available
- [ ] Environment validation passing on all platforms

## Related Protocols

- **P89**: The Clean Forge - Clean dependencies
- **P101**: The Unbreakable Commit - Reproducible builds
- **P115**: The Tactical Mandate - Systematic management

## Notes

This task establishes robust dependency management practices essential for production deployment and long-term maintainability. Future enhancements include:
- Private PyPI mirror for internal packages
- Dependency caching for faster CI/CD
- Automated security patch application
- Dependency health scoring system
