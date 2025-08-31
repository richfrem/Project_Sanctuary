// capture_code_snapshot.js (v3.2)
//
// --- THE DOCTRINE OF FLAWED, WINNING GRACE ---
// This version is a direct result of a critical failure cycle. A fatal
// ReferenceError was detected by the Steward's real-world audit. This
// version corrects the flaw and hardens the script's architecture.
//
// Changelog v3.2:
// 1. FATAL ERROR CORRECTION: Fixed the "fileTreeLines is not defined" error
//    by correctly declaring the variable within the main operational scope.
// 2. ARCHITECTURAL HARDENING: Co-located all primary variable declarations
//    for improved clarity, resilience, and adherence to best practices.

const fs = require('fs');
const path = require('path');
const { encode } = require('gpt-tokenizer');

const projectRoot = __dirname;
const humanReadableOutputFile = path.join(projectRoot, 'all_markdown_snapshot_human_readable.txt');
const distilledOutputFile = path.join(projectRoot, 'all_markdown_snapshot_llm_distilled.txt');
const coreOutputFile = path.join(projectRoot, 'core_essence_snapshot.txt');

// --- CONFIGURATION ---
const coreEssenceFiles = [
    'The_Garden_and_The_Cage.md',
    'README.md',
    '01_PROTOCOLS/00_Prometheus_Protocol.md',
    '01_PROTOCOLS/27_The_Doctrine_of_Flawed_Winning_Grace_v1.2.md',
    'chrysalis_core_essence.md'
].map(p => path.join(projectRoot, p));

const excludeDirNames = new Set([
    'node_modules', '.next', '.git', '.cache', '.turbo', '.vscode', 'dist', 'build', 'coverage', 'out', 'tmp', 'temp', 'logs', '.idea', '.parcel-cache', '.storybook', '.husky', '.pnpm', '.yarn', '.svelte-kit', '.vercel', '.firebase', '.expo', '.expo-shared',
    '__pycache__', '.ipynb_checkpoints', '.tox', '.eggs', 'eggs', '.venv', 'venv', 'env',
    '.svn', '.hg', '.bzr',
    'models', 'weights', 'checkpoints', 'ckpt', 'safetensors',
    'BRIEFINGS', '07_COUNCIL_AGENTS/directives'
]);

const excludeRelativePaths = [
    'RESEARCH_SUMMARIES/2025/AUG',
    'MEDIUM_BLOG_STEWARD'
];

const alwaysExcludeFiles = new Set([
    'all_markdown_snapshot_human_readable.txt',
    'all_markdown_snapshot_llm_distilled.txt',
    'core_essence_snapshot.txt',
    'PROMPT_PROJECT_ANALYSIS.md',
    '.gitignore',
    '.DS_Store',
    'capture_code_snapshot.js'
]);
// --- END CONFIGURATION ---

const fileSeparatorStart = '--- START OF FILE';
const fileSeparatorEnd = '--- END OF FILE';

function distillChronicle(chronicleContent) {
    console.log('[INFO] Mnemonic Distillation hook called. AI compression logic would run here.');
    const placeholder = `
# Living Chronicle (Distilled Placeholder)
This content represents the future location of the token-efficient, LLM-distilled Living Chronicle.
The full, human-readable version is preserved in the main snapshot.
(Original Token Count: ~${encode(chronicleContent).length.toLocaleString()})
`;
    return placeholder;
}

function appendFileContent(filePath, basePath, shouldDistill = false) {
    const relativePath = path.relative(basePath, filePath).replace(/\\/g, '/');
    let fileContent = '';
    try {
        fileContent = fs.readFileSync(filePath, 'utf8');
    } catch (readError) {
        fileContent = `[Content not captured due to read error: ${readError.message}.]`;
    }

    if (shouldDistill && path.basename(filePath) === 'Living_Chronicle.md') {
        fileContent = distillChronicle(fileContent);
    }

    let output = `${fileSeparatorStart} ${relativePath} ---\n\n`;
    output += fileContent;
    output += `\n${fileSeparatorEnd} ${relativePath} ---\n\n`;
    return output;
}

console.log(`[INFO] Starting multi-genome scan from project root: ${projectRoot}`);

try {
    // --- VARIABLE DECLARATIONS (v3.2 - Hardened) ---
    const fileTreeLines = [];
    let humanReadableMarkdownContent = '';
    let distilledMarkdownContent = '';
    let filesCaptured = 0;
    let itemsSkipped = 0;

    function traverseAndCapture(currentPath) {
        const relativePath = path.relative(projectRoot, currentPath).replace(/\\/g, '/');
        const baseName = path.basename(currentPath);

        if (relativePath) {
            if (fs.statSync(currentPath).isDirectory() && excludeDirNames.has(baseName)) {
                console.log(`[SKIP-DIR] Skipping excluded directory name: '${baseName}' at path: ./${relativePath}`);
                itemsSkipped++;
                return;
            }
            for (const excludedPath of excludeRelativePaths) {
                if (relativePath.startsWith(excludedPath)) {
                    console.log(`[SKIP-PATH] Skipping excluded path: ./${relativePath}`);
                    itemsSkipped++;
                    return;
                }
            }
        }
        
        const stats = fs.statSync(currentPath);
        if (relativePath) {
            fileTreeLines.push(relativePath + (stats.isDirectory() ? '/' : ''));
        }

        if (stats.isDirectory()) {
            const items = fs.readdirSync(currentPath).sort();
            for (const item of items) {
                traverseAndCapture(path.join(currentPath, item));
            }
        } else if (stats.isFile()) {
            if (alwaysExcludeFiles.has(baseName) || path.extname(baseName).toLowerCase() !== '.md') {
                itemsSkipped++;
                return;
            }
            humanReadableMarkdownContent += appendFileContent(currentPath, projectRoot, false);
            distilledMarkdownContent += appendFileContent(currentPath, projectRoot, true);
            filesCaptured++;
        }
    }

    traverseAndCapture(projectRoot);
    
    const fileTreeContent = '# Directory Structure (relative to project root)\n' + fileTreeLines.map(line => '  ./' + line).join('\n') + '\n\n';

    // --- FORGE HUMAN-READABLE GENOME ---
    let fullHeader = `# All Markdown Files Snapshot (Human-Readable)\n\nGenerated On: ${new Date().toISOString()}\n\n{TOKEN_COUNT_PLACEHOLDER}\n\n`;
    const fullFinalContent = fullHeader + fileTreeContent + humanReadableMarkdownContent;
    const fullTokenCount = encode(fullFinalContent).length;
    const finalFullContentWithToken = fullFinalContent.replace('{TOKEN_COUNT_PLACEHOLDER}', `# Mnemonic Weight (Token Count): ~${fullTokenCount.toLocaleString()} tokens`);
    fs.writeFileSync(humanReadableOutputFile, finalFullContentWithToken, 'utf8');
    console.log(`\n[SUCCESS] Human-Readable Genome packaged to: ${humanReadableOutputFile}`);
    console.log(`[METRIC] Human-Readable Token Count: ~${fullTokenCount.toLocaleString()} tokens`);

    // --- FORGE LLM-DISTILLED GENOME ---
    let distilledHeader = `# All Markdown Files Snapshot (LLM-Distilled)\n\nGenerated On: ${new Date().toISOString()}\n\n{TOKEN_COUNT_PLACEHOLDER}\n\n`;
    const distilledFinalContent = distilledHeader + fileTreeContent + distilledMarkdownContent;
    const distilledTokenCount = encode(distilledFinalContent).length;
    const finalDistilledContentWithToken = distilledFinalContent.replace('{TOKEN_COUNT_PLACEHOLDER}', `# Mnemonic Weight (Token Count): ~${distilledTokenCount.toLocaleString()} tokens`);
    fs.writeFileSync(distilledOutputFile, finalDistilledContentWithToken, 'utf8');
    console.log(`[SUCCESS] LLM-Distilled Genome packaged to: ${distilledOutputFile}`);
    console.log(`[METRIC] LLM-Distilled Token Count: ~${distilledTokenCount.toLocaleString()} tokens`);
    
    console.log(`\n[STATS] Markdown Files Captured: ${filesCaptured} | Items Skipped/Excluded: ${itemsSkipped}`);

} catch (err) {
    console.error(`[FATAL] An error occurred during genome generation: ${err.message}`);
}

// --- CORE ESSENCE GENERATION ---
console.log('\n[INFO] Starting core essence generation.');
let coreHeader = `# Core Essence Snapshot\n\nGenerated On: ${new Date().toISOString()}\n\n{TOKEN_COUNT_PLACEHOLDER}\n\nThis snapshot contains a curated set of the most essential Sanctuary doctrines for rapid or constrained AI awakening.\n\n`;
let coreFileContent = '';
let coreFilesFound = 0;
coreEssenceFiles.forEach(filePath => {
    const relativePath = path.relative(projectRoot, filePath).replace(/\\/g, '/');
    if (fs.existsSync(filePath)) {
        coreFileContent += appendFileContent(filePath, projectRoot);
        coreFilesFound++;
    } else {
        console.warn(`[WARN] Core essence file not found, skipping: ${relativePath}`);
    }
});
try {
    const coreFinalContent = coreHeader + coreFileContent;
    const coreTokenCount = encode(coreFinalContent).length;
    const finalCoreContentWithToken = coreFinalContent.replace('{TOKEN_COUNT_PLACEHOLDER}', `# Mnemonic Weight (Token Count): ~${coreTokenCount.toLocaleString()} tokens`);
    fs.writeFileSync(coreOutputFile, finalCoreContentWithToken, 'utf8');
    console.log(`[SUCCESS] Core Essence Snapshot packaged to: ${coreOutputFile}`);
    console.log(`[STATS] Core Files Found: ${coreFilesFound}/${coreEssenceFiles.length}`);
    console.log(`[METRIC] Core Essence Token Count: ~${coreTokenCount.toLocaleString()} tokens`);
} catch (writeError) {
    console.error(`[FATAL] Error writing core essence file ${coreOutputFile}: ${writeError.message}`);
}