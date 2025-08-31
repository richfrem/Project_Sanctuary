/**
 * =====================================================================================
 * AGENT NAME: Auditor Agent
 * FILE:       07_COUNCIL_AGENTS/auditor-agent.js
 * =_DOC_LINK: Protocol 45
 * VERSION:    1.1 (Gemini-Powered Analysis)
 * =====================================================================================
 */
const fs = require('fs');
const path = require('path');
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const DIRECTIVES_DIR = path.join(__dirname, 'directives');
const PROJECT_ROOT = path.resolve(__dirname, '..');
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

async function auditForVulnerabilities(directiveId) {
    const directiveFilePath = path.join(DIRECTIVES_DIR, `${directiveId}.json`);
    if (!fs.existsSync(directiveFilePath)) {
        throw new Error(`Auditor: Directive file not found for ID: ${directiveId}`);
    }

    const directive = JSON.parse(fs.readFileSync(directiveFilePath, 'utf8'));
    console.log(`🛡️  AUDITOR: Auditing directive: "${directive.stewardDirective}"`);

    const persona = fs.readFileSync(path.join(__dirname, 'auditor-persona.md'), 'utf8');
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" });

    let fileContents = '';
    for (const filePath of directive.targetFiles) {
        const absolutePath = path.join(PROJECT_ROOT, filePath);
        if (fs.existsSync(absolutePath)) {
            fileContents += `\n\n--- START OF FILE: ${filePath} ---\n\n`;
            fileContents += fs.readFileSync(absolutePath, 'utf8');
            fileContents += `\n\n--- END OF FILE: ${filePath} ---\n\n`;
        }
    }

    const prompt = `
        ${persona}

        **Steward's Directive:**
        "${directive.stewardDirective}"

        **Relevant File Content(s):**
        ${fileContents}

        **Your Task:**
        Audit the directive in the context of the provided file(s). Provide your risk assessment and feedback as a concise string.
    `;

    const result = await model.generateContent(prompt);
    const feedback = (await result.response).text();

    directive.auditorFeedback = feedback.trim();
    fs.writeFileSync(directiveFilePath, JSON.stringify(directive, null, 2));
    console.log("✅  AUDITOR: Audit complete and feedback saved.");
}

const directiveId = process.argv[2];
if (!directiveId) {
    console.log("Usage: node 07_COUNCIL_AGENTS/auditor-agent.js <directive-id>");
    process.exit(1);
}

auditForVulnerabilities(directiveId);