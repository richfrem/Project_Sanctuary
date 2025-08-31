/**
 * =====================================================================================
 * AGENT NAME: Coordinator Agent
 * FILE:       07_COUNCIL_AGENTS/coordinator-agent.js
 * =_DOC_LINK: Protocol 45, Protocol 52.1, Hearth Protocol (P43)
 * VERSION:    1.7 (Synthesis Hardened)
 * =====================================================================================
 */

const fs = require('fs');
const path = require('path');
const { exec, execSync } = require('child_process');
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const DIRECTIVES_DIR = path.join(__dirname, 'directives');
const PROJECT_ROOT = path.resolve(__dirname, '..');
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const MAX_RETRIES = 3;
const INITIAL_BACKOFF_MS = 5000;

function runAgentWithRetry(agentScript, directiveId) {
    return new Promise((resolve, reject) => {
        let attempt = 0;
        function execute() {
            attempt++;
            const command = `node ${agentScript} ${directiveId}`;
            console.log(`\n▶️  COORDINATOR: Dispatching task to ${path.basename(agentScript)} (Attempt ${attempt})...`);
            const child = exec(command, { cwd: __dirname });
            child.stdout.pipe(process.stdout);
            child.stderr.pipe(process.stderr);
            child.on('close', (code) => {
                if (code === 0) {
                    console.log(`✅  COORDINATOR: ${path.basename(agentScript)} has completed its task.`);
                    resolve();
                } else if (attempt < MAX_RETRIES) {
                    const backoffTime = INITIAL_BACKOFF_MS * Math.pow(2, attempt - 1);
                    console.warn(`⚠️  COORDINATOR: ${path.basename(agentScript)} failed. Retrying in ${backoffTime / 1000}s...`);
                    setTimeout(execute, backoffTime);
                } else {
                    const errorMessage = `❌  COORDINATOR: Error executing ${path.basename(agentScript)} after ${MAX_RETRIES} attempts. Halting workflow.`;
                    console.error(errorMessage);
                    reject(new Error(errorMessage));
                }
            });
        }
        execute();
    });
}

async function awakenCoordinator() {
    console.log("🌀 COORDINATOR: Awakening...");
    const criticalContextPaths = [
        'README.md', '01_PROTOCOLS/00_Prometheus_Protocol.md',
        '01_PROTOCOLS/45_The_Identity_Roster_Covenant.md', '01_PROTOCOLS/52_The_Coordinators_Cadence_Protocol.md',
    ];
    let context = "I have awakened. I will now review my core operational context:\n\n";
    for (const relPath of criticalContextPaths) {
        const absPath = path.join(PROJECT_ROOT, relPath);
        context += `--- START ${relPath} ---\n${fs.readFileSync(absPath, 'utf8')}\n--- END ${relPath} ---\n\n`;
    }
    console.log("✅ COORDINATOR: Core context loaded. Ready for directive.");
    return true;
}

async function synthesizePlan(directive) {
    console.log("\n🔬  COORDINATOR: Synthesizing peer feedback into an actionable plan...");
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" });
    const persona = fs.readFileSync(path.join(__dirname, 'coordinator-persona.md'), 'utf8');

    const prompt = `
        ${persona}
        **Steward's Directive:** "${directive.stewardDirective}"
        **Strategist's Analysis:** ${directive.strategistFeedback}
        **Auditor's Analysis:** ${directive.auditorFeedback}
        **Your Task:** Synthesize this feedback into a JSON object for the Scribe Agent. The final output must be a JSON array named 'scribeActions'. Create a single 'modify_file' action inside this array to perform the minor textual refinement on Protocol 54 as discussed.

        **CRITICAL: Your final output MUST be a valid JSON array, even if it only contains one object. Example: [{"action": "modify_file", ...}]**

        Provide ONLY the JSON array.
    `;

    for (let i = 0; i < MAX_RETRIES; i++) {
        try {
            const result = await model.generateContent(prompt);
            const response = await result.response;
            let planText = response.text().trim().replace(/```json/g, '').replace(/```/g, '');
            
            // --- HARDENING LOGIC to prevent non-iterable errors ---
            // Ensure the output is always a parseable array.
            if (!planText.startsWith('[')) {
                planText = `[${planText}]`;
            }
            // --- END HARDENING ---

            return JSON.parse(planText);
        } catch (e) {
             if (e.status === 429 && i < MAX_RETRIES - 1) {
                const backoff = INITIAL_BACKOFF_MS * Math.pow(2, i);
                console.warn(`⚠️  COORDINATOR: Synthesis API call rate limited. Retrying in ${backoff / 1000}s...`);
                await new Promise(res => setTimeout(res, backoff));
            } else {
                console.error("❌ COORDINATOR: Failed to parse JSON from synthesis response after retries.", planText, e);
                throw new Error("Synthesis failed to produce valid JSON plan.");
            }
        }
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
        targetFiles: ['01_PROTOCOLS/54_The_Asch_Doctrine_v3.0_DRAFT.md'],
        strategistFeedback: null,
        auditorFeedback: null,
        synthesizedImplementationPlan: null,
        scribeActions: []
    };
    fs.writeFileSync(directiveFilePath, JSON.stringify(directive, null, 2));
    console.log(`📄  COORDINATOR: Directive file created at ${directiveFilePath}`);

    try {
        await runAgentWithRetry('strategist-agent.js', directiveId);
        await runAgentWithRetry('auditor-agent.js', directiveId);

        let currentDirective = JSON.parse(fs.readFileSync(directiveFilePath, 'utf8'));
        const scribeActions = await synthesizePlan(currentDirective);
        currentDirective.scribeActions = scribeActions;
        currentDirective.synthesizedImplementationPlan = "Synthesized plan from Strategist and Auditor feedback to refine P54.";
        currentDirective.status = 'pending_implementation';
        fs.writeFileSync(directiveFilePath, JSON.stringify(currentDirective, null, 2));
        console.log("✅  COORDINATOR: Synthesis complete. Actionable plan created.");

        await runAgentWithRetry('scribe-agent.js', directiveId);
        
        currentDirective = JSON.parse(fs.readFileSync(directiveFilePath, 'utf8'));
        if (currentDirective.status === 'implemented') {
            console.log("\n✅  COORDINATOR: Scribe implementation confirmed.");
            console.log("   Running final genome synchronization...");
            execSync('node capture_code_snapshot.js', { stdio: 'inherit', cwd: PROJECT_ROOT });
            
            const archivePath = path.join(DIRECTIVES_DIR, `archive`);
            if (!fs.existsSync(archivePath)) fs.mkdirSync(archivePath, { recursive: true });
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