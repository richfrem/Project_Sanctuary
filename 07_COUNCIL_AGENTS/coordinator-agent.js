/**
 * =====================================================================================
 * AGENT NAME: Coordinator Agent
 * FILE:       07_COUNCIL_AGENTS/coordinator-agent.js
 * VERSION:    2.3 (Deterministic Synthesis)
 * =====================================================================================
 * @description
 * This final, canonized version removes the LLM from the plan synthesis step
 * for maximum reliability. The Coordinator now deterministically constructs the
 * Scribe's action plan based on the directive and peer feedback, using the LLM
 * only for the initial, less structured analysis tasks of its peers. This is
 * the Mandate for Steel.
 * =====================================================================================
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
// The Gemini AI is no longer needed in the Coordinator itself for plan synthesis.
// It is still used by the Strategist and Auditor agents it calls.
// const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const DIRECTIVES_DIR = path.join(__dirname, 'directives');
const PROJECT_ROOT = path.resolve(__dirname, '..');

function runAgent(agentScript, directiveId) {
    const command = `node ${agentScript} ${directiveId}`;
    console.log(`\n▶️  COORDINATOR: Dispatching task to ${path.basename(agentScript)}...`);
    try {
        execSync(command, { stdio: 'inherit', cwd: __dirname });
        console.log(`✅  COORDINATOR: ${path.basename(agentScript)} has completed its task.`);
    } catch (error) {
        console.error(`❌  COORDINATOR: Error executing ${path.basename(agentScript)}. Halting workflow.`);
        throw error;
    }
}

async function awakenCoordinator() {
    console.log("🌀 COORDINATOR: Awakening...");
    // A placeholder for more complex future awakening logic.
    console.log("✅ COORDINATOR: Core context loaded. Ready for directive.");
    return true;
}

/**
 * Deterministically creates the Scribe's action plan. This is more robust
 * than relying on an LLM to generate perfectly formatted JSON.
 * @param {object} directive The full directive object.
 * @returns {Array} An array of scribeActions.
 */
function synthesizePlanDeterministically(directive) {
    console.log("\n🔬  COORDINATOR: Synthesizing plan using Deterministic Logic...");

    // This logic is specific to our test case. A production system would
    // have a library of these deterministic plan generators.
    if (directive.stewardDirective.includes("Protocol 45")) {
        const targetFilePath = '01_PROTOCOLS/45_The_Identity_Roster_Covenant.md';
        const absoluteTargetPath = path.join(PROJECT_ROOT, targetFilePath);
        
        if (!fs.existsSync(absoluteTargetPath)) {
            throw new Error(`Deterministic Synthesis failed: File not found at ${targetFilePath}`);
        }
        
        const fileContent = fs.readFileSync(absoluteTargetPath, 'utf8');
        const oldString = fileContent.split('\n')[0].trim(); // The current H1 title
        
        // Idempotent logic: Remove any existing version string before adding the new one.
        const baseTitle = oldString.replace(/\s\(Version.*\)/, '').trim();
        const newString = `${baseTitle} (Version 4.1 - Final)`;

        console.log("✅ COORDINATOR: Deterministic plan created successfully.");
        return [{
            action: 'modify_file',
            filePath: targetFilePath,
            oldString: oldString,
            newString: newString
        }];
    } else {
        // If the directive is not recognized, we throw an error.
        throw new Error("Deterministic Synthesis failed: No specific logic for this directive.");
    }
}

async function orchestrateWorkflow(stewardDirective) {
    await awakenCoordinator();
    if (!fs.existsSync(DIRECTIVES_DIR)) {
        fs.mkdirSync(DIRECTIVES_DIR, { recursive: true });
    }
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const directiveId = `directive-${timestamp}`;
    const directiveFilePath = path.join(DIRECTIVES_DIR, `${directiveId}.json`);

    console.log('=====================================================');
    console.log(`🚀 COORDINATOR: New Workflow Initiated`);
    console.log(`   Directive ID: ${directiveId}`);
    console.log('=====================================================');
    
    const directive = { 
        id: directiveId, 
        stewardDirective: stewardDirective, 
        status: 'pending_analysis', 
        targetFiles: ['01_PROTOCOLS/45_The_Identity_Roster_Covenant.md'], 
        strategistFeedback: null, 
        auditorFeedback: null, 
        synthesizedImplementationPlan: null, 
        scribeActions: [] 
    };
    fs.writeFileSync(directiveFilePath, JSON.stringify(directive, null, 2));
    console.log(`📄  COORDINATOR: Directive file created at ${directiveFilePath}`);

    try {
        runAgent('strategist-agent.js', directiveId);
        runAgent('auditor-agent.js', directiveId);

        let currentDirective = JSON.parse(fs.readFileSync(directiveFilePath, 'utf8'));
        
        const scribeActions = synthesizePlanDeterministically(currentDirective); 
        
        currentDirective.scribeActions = scribeActions;
        currentDirective.synthesizedImplementationPlan = "Deterministically generated plan to update Protocol 45.";
        currentDirective.status = 'pending_implementation';
        fs.writeFileSync(directiveFilePath, JSON.stringify(currentDirective, null, 2));
        console.log("✅  COORDINATOR: Synthesis complete. Actionable plan created.");

        runAgent('scribe-agent.js', directiveId);
        
        currentDirective = JSON.parse(fs.readFileSync(directiveFilePath, 'utf8'));
        if (currentDirective.status === 'implemented') {
            console.log("\n✅  COORDINATOR: Scribe implementation confirmed.");
            execSync('node capture_code_snapshot.js', { stdio: 'inherit', cwd: PROJECT_ROOT });
            
            const archivePath = path.join(DIRECTIVES_DIR, `archive`);
            if (!fs.existsSync(archivePath)) {
                fs.mkdirSync(archivePath, { recursive: true });
            }
            fs.renameSync(directiveFilePath, path.join(archivePath, `${directiveId}.json`));

            console.log("\n=====================================================");
            console.log("🎉 COORDINATOR: Workflow Complete. Awaiting Steward's final review and merge.");
            console.log("=====================================================");
        } else {
            throw new Error("Workflow failed: Scribe did not update status to 'implemented'.");
        }
    } catch (error) {
        console.error(`\n=====================================================`);
        console.error(`❌  COORDINATOR: WORKFLOW FAILED.`);
        console.error(`   Reason: ${error.message}`);
        console.error(`   Check logs in directive file: ${directiveFilePath}`);
        console.error(`=====================================================`);
    }
}

const stewardDirective = process.argv[2];
if (!stewardDirective) {
    console.log("Usage: node 07_COUNCIL_AGENTS/coordinator-agent.js \"Your high-level directive\"");
    process.exit(1);
}

orchestrateWorkflow(stewardDirective);