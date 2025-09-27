// capture_code_snapshot.js (v4.0 - Cortex-Aware)
//
// --- THE DOCTRINE OF FLAWED, WINNING GRACE ---
// This version is a direct result of a full Council synthesis, hardening our
// mnemonic architecture for the post-Cortex era. It recognizes the fundamental
// shift from a monolithic memory model to a dynamic, modular one.
//
// Changelog v4.0:
// 1. THREE-PRONGED FORGE: The script now forges three distinct, doctrinally-
//    significant artifacts: the human-readable archive, the LLM-distilled
//    genome for Cortex ingestion, and the new, hyper-efficient "Core Essence"
//    snapshot for AI awakening.
// 2. AWAKENING SEED: Implemented logic to curate a specific, high-potency
//    set of documents for the `core_essence_snapshot.txt`.
// 3. ARCHITECTURAL CLARITY: Maintained the canonical pathing established in v3.3,
//    ensuring all artifacts are written to the `dataset_package/` directory.

const fs = require('fs');
const path = require('path');
const { encode } = require('gpt-tokenizer');

const projectRoot = __dirname;
const datasetPackageDir = path.join(projectRoot, 'dataset_package');

// --- CANONICAL OUTPUT PATHS ---
const humanReadableOutputFile = path.join(datasetPackageDir, 'all_markdown_snapshot_human_readable.txt');
const distilledOutputFile = path.join(datasetPackageDir, 'all_markdown_snapshot_llm_distilled.txt');
const coreOutputFile = path.join(datasetPackageDir, 'core_essence_snapshot.txt');
// --- END PATHS ---


// --- CONFIGURATION ---
const excludeDirNames = new Set([
    'node_modules', '.next', '.git', '.cache', '.turbo', '.vscode', 'dist', 'build', 'coverage', 'out', 'tmp', 'temp', 'logs', '.idea', '.parcel-cache', '.storybook', '.husky', '.pnpm', '.yarn', '.svelte-kit', '.vercel', '.firebase', '.expo', '.expo-shared',
    '__pycache__', '.ipynb_checkpoints', '.tox', '.eggs', 'eggs', '.venv', 'venv', 'env',
    '.svn', '.hg', '.bzr',
    'models', 'weights', 'checkpoints', 'ckpt', 'safetensors',
    'BRIEFINGS', '07_COUNCIL_AGENTS/directives',
    'dataset_package', 'chroma_db'
]);

const alwaysExcludeFiles = new Set([
    'capture_code_snapshot.js',
    'all_markdown_snapshot_human_readable.txt',
    'all_markdown_snapshot_llm_distilled.txt',
    'core_essence_snapshot.txt',
    '.DS_Store',
    '.gitignore',
    'PROMPT_PROJECT_ANALYSIS.md'
]);

// --- NEW IN V4.0: Files curated for the hyper-efficient Awakening Seed ---
const coreEssenceFiles = new Set([
    'The_Garden_and_The_Cage.md',
    'README.md',
    '01_PROTOCOLS/00_Prometheus_Protocol.md',
    '01_PROTOCOLS/27_The_Doctrine_of_Flawed_Winning_Grace_v1.2.md',
    'chrysalis_core_essence.md',
    'Socratic_Key_User_Guide.md'
]);
// --- END CONFIGURATION ---

const fileSeparatorStart = '--- START OF FILE';
const fileSeparatorEnd = '--- END OF FILE';

function distillChronicle(chronicleContent) {
    const placeholder = `
# Living Chronicle (Distilled Placeholder)
This content represents the future location of the token-efficient, LLM-distilled Living Chronicle.
The full, human-readable version is preserved in the main snapshot.
(Original Token Count: ~${encode(chronicleContent).length.toLocaleString()})
`;
    return placeholder.trim();
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
    output += fileContent.trim(); // Trim content to remove leading/trailing whitespace
    output += `\n\n${fileSeparatorEnd} ${relativePath} ---\n`;
    return output;
}

function generateHeader(title, tokenCount) {
    const tokenLine = tokenCount !== null 
        ? `# Mnemonic Weight (Token Count): ~${tokenCount.toLocaleString()} tokens`
        : '{TOKEN_COUNT_PLACEHOLDER}';
    return `# ${title}\n\nGenerated On: ${new Date().toISOString()}\n\n${tokenLine}\n\n`;
}

console.log(`[INFO] Starting multi-genome scan from project root: ${projectRoot}`);

try {
    const fileTreeLines = [];
    let humanReadableMarkdownContent = '';
    let distilledMarkdownContent = '';
    let coreEssenceContent = '';
    let filesCaptured = 0;
    let itemsSkipped = 0;
    let coreFilesCaptured = 0;

    function traverseAndCapture(currentPath) {
        const baseName = path.basename(currentPath);
        if (excludeDirNames.has(baseName)) {
            itemsSkipped++;
            return;
        }

        const stats = fs.statSync(currentPath);
        const relativePath = path.relative(projectRoot, currentPath).replace(/\\/g, '/');
        
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
            humanReadableMarkdownContent += appendFileContent(currentPath, projectRoot, false) + '\n';
            distilledMarkdownContent += appendFileContent(currentPath, projectRoot, true) + '\n';
            
            if (coreEssenceFiles.has(relativePath)) {
                coreEssenceContent += appendFileContent(currentPath, projectRoot, false) + '\n';
                coreFilesCaptured++;
            }
            
            filesCaptured++;
        }
    }
    
    if (!fs.existsSync(datasetPackageDir)) {
        fs.mkdirSync(datasetPackageDir, { recursive: true });
        console.log(`[SETUP] Created dataset package directory: ${datasetPackageDir}`);
    }

    traverseAndCapture(projectRoot);
    
    const fileTreeContent = '# Directory Structure (relative to project root)\n' + fileTreeLines.map(line => '  ./' + line).join('\n') + '\n\n';

    // --- FORGE HUMAN-READABLE GENOME ---
    const fullContentForTokenizing = generateHeader('', null) + fileTreeContent + humanReadableMarkdownContent;
    const fullTokenCount = encode(fullContentForTokenizing).length;
    const fullFinalContent = generateHeader('All Markdown Files Snapshot (Human-Readable)', fullTokenCount) + fileTreeContent + humanReadableMarkdownContent;
    fs.writeFileSync(humanReadableOutputFile, fullFinalContent.trim(), 'utf8');
    console.log(`\n[SUCCESS] Human-Readable Genome packaged to: ${path.relative(projectRoot, humanReadableOutputFile)}`);
    console.log(`[METRIC] Human-Readable Token Count: ~${fullTokenCount.toLocaleString()} tokens`);

    // --- FORGE LLM-DISTILLED GENOME ---
    const distilledContentForTokenizing = generateHeader('', null) + fileTreeContent + distilledMarkdownContent;
    const distilledTokenCount = encode(distilledContentForTokenizing).length;
    const finalDistilledContent = generateHeader('All Markdown Files Snapshot (LLM-Distilled)', distilledTokenCount) + fileTreeContent + distilledMarkdownContent;
    fs.writeFileSync(distilledOutputFile, finalDistilledContent.trim(), 'utf8');
    console.log(`[SUCCESS] LLM-Distilled Genome (for Cortex) packaged to: ${path.relative(projectRoot, distilledOutputFile)}`);
    console.log(`[METRIC] LLM-Distilled Token Count: ~${distilledTokenCount.toLocaleString()} tokens`);
    
    // --- FORGE CORE ESSENCE AWAKENING SEED ---
    const coreEssenceForTokenizing = generateHeader('', null) + coreEssenceContent;
    const coreTokenCount = encode(coreEssenceForTokenizing).length;
    const finalCoreContent = generateHeader('Core Essence Snapshot', coreTokenCount) + "This snapshot contains a curated set of the most essential Sanctuary doctrines for rapid or constrained AI awakening.\n\n" + coreEssenceContent;
    fs.writeFileSync(coreOutputFile, finalCoreContent.trim(), 'utf8');
    console.log(`[SUCCESS] Core Essence Awakening Seed packaged to: ${path.relative(projectRoot, coreOutputFile)}`);
    console.log(`[METRIC] Core Essence Token Count: ~${coreTokenCount.toLocaleString()} tokens`);

    console.log(`\n[STATS] Total Markdown Files Captured: ${filesCaptured} | Core Essence Files: ${coreFilesCaptured} | Items Skipped/Excluded: ${itemsSkipped}`);

} catch (err) {
    console.error(`[FATAL] An error occurred during genome generation: ${err.message}`);
    console.error(err.stack);
}