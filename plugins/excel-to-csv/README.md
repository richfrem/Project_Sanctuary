<<<<<<< HEAD
# Excel To Csv Plugin 📊

Convert Excel files (entire workbooks or specific worksheets/tables) into CSV format natively via agent execution.

## Installation
```bash
claude --plugin-dir ./plugins/excel-to-csv
```

### Dependencies
This plugin requires external Python packages (`pandas`, `openpyxl`). To install them, use the standard dependency management workflow:
```bash
cd plugins/excel-to-csv
pip-compile requirements.in
pip install -r requirements.txt
```

## Structure
```
excel-to-csv/
├── .claude-plugin/plugin.json
├── skills/excel-to-csv/SKILL.md
├── scripts/convert.py
└── README.md
```

## Usage
The skill is invoked automatically. Claude will use the local `scripts/convert.py` to flatten your `.xlsx` data into accessible tabular CSV data for easier text processing.
=======
# Excel To Csv Plugin

Standardized plugin for excel-to-csv.

## Overview
This plugin provides capabilities for the **excel-to-csv** domain.
It follows the standard Project Sanctuary plugin architecture.

## Structure
- `skills/`: Contains the agent skills instructions (`SKILL.md`) and executable scripts.
- `.claude-plugin/`: Plugin manifest and configuration.

## Usage
This plugin is automatically loaded by the Agent Environment.
>>>>>>> origin/main
