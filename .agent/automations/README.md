# Sanctuary Bot automations

One JSON definition per automation (`<name>.json`), reviewed and merged like code. A definition does nothing until it is approved on the machine that runs it (`bot.py approve <name> --by <approver>`), and any later edit pauses it until it is approved again.

Set up, run and manage automations with the `operating-sanctuary-bot` skill (`plugins/sanctuary-core/.apm/skills/operating-sanctuary-bot/`); format: its `references/automations.md`. Run state, history and results stay machine-local in `~/.sanctuary/daemon/`.
