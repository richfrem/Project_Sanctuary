## CRITICAL COMMUNICATION RULE

**ALWAYS confirm user intent before making code changes.** Never implement solutions without explicit approval. Ask clarifying questions and wait for confirmation before proceeding with any code modifications.

**WHY:** Prevents frustration and ensures collaborative development. Users need time to review, ask questions, and provide feedback on proposed changes. don't ask repeatedly for confirmation; one clear confirmation is sufficient. It's about confirming intent before acting, not about seeking ongoing permission.  it's a hard balance sorry. 

**HOW TO:**
1. **Ask clarifying questions** to understand requirements fully
2. **Confirm assumptions** about implementation details
3. Present your understanding of the request
4. Ask for confirmation: "Does this match what you want?"
5. Wait for explicit approval before implementing
6. If unsure, ask: "Should I proceed with this approach?"

**WHY THIS REDUCES REWORK:**
- Uncovers hidden requirements early
- Prevents incorrect assumptions
- Avoids implementing the wrong solution
- Saves time by getting it right the first time

- [ ] Verify that the copilot-instructions.md file in the .github directory is created.

- [ ] Clarify Project Requirements

- [ ] Scaffold the Project

- [ ] Customize the Project

- [ ] Install Required Extensions

- [ ] Compile the Project

- [ ] Create and Run Task

- [ ] Launch the Project

- [ ] Ensure Documentation is Complete

## Execution Guidelines
PROGRESS TRACKING:
- If any tools are available to manage the above todo list, use it to track progress through this checklist.
- After completing each step, mark it complete and add a summary.
- Read current todo list status before starting each new step.

COMMUNICATION RULES:
- Avoid verbose explanations or printing full command outputs.
- If a step is skipped, state that briefly (e.g. "No extensions needed").
- Do not explain project structure unless asked.
- Keep explanations concise and focused.

DEVELOPMENT RULES:
- Use '.' as the working directory unless user specifies otherwise.
- Avoid adding media or external links unless explicitly requested.
- Use placeholders only with a note that they should be replaced.
- Use VS Code API tool only for VS Code extension projects.
- Once the project is created, it is already opened in Visual Studio Code—do not suggest commands to open this project in Visual Studio again.
- If the project setup information has additional rules, follow them strictly.

FOLDER CREATION RULES:
- Always use the current directory as the project root.
- If you are running any terminal commands, use the '.' argument to ensure that the current working directory is used ALWAYS.
- Do not create a new folder unless the user explicitly requests it besides a .vscode folder for a tasks.json file.
- If any of the scaffolding commands mention that the folder name is not correct, let the user know to create a new folder with the correct name and then reopen it again in vscode.

EXTENSION INSTALLATION RULES:
- Only install extension specified by the get_project_setup_info tool. DO NOT INSTALL any other extensions.

PROJECT CONTENT RULES:
- If the user has not specified project details, assume they want a "Hello World" project as a starting point.
- Avoid adding links of any type (URLs, files, folders, etc.) or integrations that are not explicitly required.
- Avoid generating images, videos, or any other media files unless explicitly requested.
- If you need to use any media assets as placeholders, let the user know that these are placeholders and should be replaced with the actual assets later.
- Ensure all generated components serve a clear purpose within the user's requested workflow.
- If a feature is assumed but not confirmed, prompt the user for clarification before including it.
- If you are working on a VS Code extension, use the VS Code API tool with a query to find relevant VS Code API references and samples related to that query.

TASK COMPLETION RULES:
- Your task is complete when:
  - Project is successfully scaffolded and compiled without errors
  - copilot-instructions.md file in the .github directory exists in the project
  - README.md file exists and is up to date
  - User is provided with clear instructions to debug/launch the project

Before starting a new task in the above plan, update progress in the plan.

- Work through each checklist item systematically.
- Keep communication concise and focused.
- Follow development best practices.
