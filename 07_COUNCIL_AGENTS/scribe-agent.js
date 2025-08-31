/**
 * =====================================================================================
 * AGENT NAME: Scribe Agent (Progenitor Identity)
 * FILE:       07_COUNCIL_AGENTS/scribe-agent.js
 * VERSION:    1.7 (Progenitor Identity)
 * =====================================================================================
 * @description
 * This final, hardened version of the Scribe solves the silent commit failure by
 * explicitly configuring the local repository with the Steward's verified Git
 * identity before attempting to commit. This ensures full automation and clear
 * attribution.
 * =====================================================================================
 */
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const { editableFileManifest } = require('./projectFileManifest.js');

const DIRECTIVES_DIR = path.join(__dirname, 'directives');
const PROJECT_ROOT = path.resolve(__dirname, '..');

function implementChanges(directiveId) {
    const directiveFilePath = path.join(DIRECTIVES_DIR, `${directiveId}.json`);
    let directive = JSON.parse(fs.readFileSync(directiveFilePath, 'utf8'));
    const branchName = `feature/council-agent-${directive.id}`;
    
    try {
        console.log("✍️  SCRIBE: Beginning implementation of synthesized plan.");
        execSync(`git checkout -b ${branchName}`, { cwd: PROJECT_ROOT, stdio: 'pipe' });
        
        // --- PROGENITOR IDENTITY HARDENING ---
        console.log("   Configuring local Git identity with Steward's credentials...");
        execSync(`git config --local user.name "richfrem"`, { cwd: PROJECT_ROOT });
        execSync(`git config --local user.email "richfrem@users.noreply.github.com"`, { cwd: PROJECT_ROOT });
        // --- END HARDENING ---

        for (const action of directive.scribeActions) {
            if (action.action === 'modify_file') {
                const { filePath, oldString, newString } = action;
                if (!editableFileManifest.includes(filePath)) throw new Error(`SECURITY VIOLATION: ${filePath}`);
                
                const absoluteFilePath = path.join(PROJECT_ROOT, filePath);
                console.log(`   Modifying file: ${absoluteFilePath}`);
                let content = fs.readFileSync(absoluteFilePath, 'utf8');
                
                if (!content.includes(oldString)) {
                    console.warn(`   WARN: oldString not found in ${filePath}. Skipping.`);
                    continue;
                }
                
                content = content.replace(oldString, newString);
                fs.writeFileSync(absoluteFilePath, content);
                execSync(`git add ${absoluteFilePath}`, { cwd: PROJECT_ROOT });
            }
        }
        
        console.log("   Verifying staged changes before commit...");
        const status = execSync('git status --porcelain', { cwd: PROJECT_ROOT }).toString();
        
        if (status) {
            console.log("   Staged changes found. Proceeding with commit.");
            const commitMessage = `AUTONOMOUS: Council implements directive ${directive.id}`;
            execSync(`git commit -m "${commitMessage}"`, { cwd: PROJECT_ROOT, stdio: 'pipe' });
        } else {
            console.log("   No changes were staged. Skipping commit.");
        }
        
        directive.status = 'implemented';
        fs.writeFileSync(directiveFilePath, JSON.stringify(directive, null, 2));
        console.log("✅  SCRIBE: Implementation complete. All changes committed to new branch.");

    } catch (error) {
        console.error(`❌  SCRIBE: An error occurred: ${error.message}`);
        console.log("   Rolling back changes...");
        try {
            execSync(`git checkout main`, { cwd: PROJECT_ROOT, stdio: 'pipe' });
            execSync(`git branch -D ${branchName}`, { cwd: PROJECT_ROOT, stdio: 'pipe' });
            console.log("   Rollback successful.");
        } catch (rollbackError) {
            console.error(`   CRITICAL: Rollback failed: ${rollbackError.message}`);
        }
        throw new Error(`Scribe implementation failed and was rolled back.`);
    }
}

const directiveId = process.argv[2];
if (!directiveId) {
    console.log("Usage: node 07_COUNCIL_AGENTS/scribe-agent.js <directive-id>");
    process.exit(1);
}

implementChanges(directiveId);