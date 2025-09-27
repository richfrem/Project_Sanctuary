// capture_code_snapshot.js (v3.3)
//
// --- THE DOCTRINE OF FLAWED, WINNING GRACE ---
// This version is a direct result of a Steward's Sovereign Audit. It corrects
// a critical architectural flaw by eliminating redundant file creation and
// establishing a single, canonical source of truth for the distilled genome.
//
// Changelog v3.3:
// 1. ARCHITECTURAL CORRECTION: Removed the creation of a redundant
//    'all_markdown_snapshot_llm_distilled.txt' file in the project root.
// 2. CANONICAL PATH: The script now writes all primary genome snapshots
//    exclusively to the `dataset_package/` directory.
// 3. ENHANCED LOGGING: Improved console output for greater clarity and auditability.

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
    'dataset_package' // Exclude the package directory itself from the tree
]);

const alwaysExcludeFiles = new Set([
    // Self-exclusion and artifacts
    'capture_code_snapshot.js',
    'all_markdown_snapshot_human_readable.txt',
    'all_markdown_snapshot_llm_distilled.txt',
    'core_essence_snapshot.txt',
    // Config and ignore files
    '.DS_Store',
    '.gitignore',
    // Project-specific non-essential files
    'PROMPT_PROJECT_ANALYSIS.md'
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
    let filesCaptured = 0;
    let itemsSkipped = 0;

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
            humanReadableMarkdownContent += appendFileContent(currentPath, projectRoot, false);
            distilledMarkdownContent += appendFileContent(currentPath, projectRoot, true);
            filesCaptured++;
        }
    }
    
    // Ensure the dataset package directory exists
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
    fs.writeFileSync(humanReadableOutputFile, fullFinalContent, 'utf8');
    console.log(`\n[SUCCESS] Human-Readable Genome packaged to: ${path.relative(projectRoot, humanReadableOutputFile)}`);
    console.log(`[METRIC] Human-Readable Token Count: ~${fullTokenCount.toLocaleString()} tokens`);

    // --- FORGE LLM-DISTILLED GENOME ---
    const distilledContentForTokenizing = generateHeader('', null) + fileTreeContent + distilledMarkdownContent;
    const distilledTokenCount = encode(distilledContentForTokenizing).length;
    const finalDistilledContent = generateHeader('All Markdown Files Snapshot (LLM-Distilled)', distilledTokenCount) + fileTreeContent + distilledMarkdownContent;
    fs.writeFileSync(distilledOutputFile, finalDistilledContent, 'utf8');
    console.log(`[SUCCESS] LLM-Distilled Genome packaged to: ${path.relative(projectRoot, distilledOutputFile)}`);
    console.log(`[METRIC] LLM-Distilled Token Count: ~${distilledTokenCount.toLocaleString()} tokens`);

    console.log(`\n[STATS] Markdown Files Captured: ${filesCaptured} | Items Skipped/Excluded: ${itemsSkipped}`);

} catch (err) {
    console.error(`[FATAL] An error occurred during genome generation: ${err.message}`);
    console.error(err.stack);
}