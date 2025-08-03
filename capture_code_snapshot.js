// capture_code_snapshot.js (v2.2)
// Changelog v2.2: Added 'BRIEFINGS' directory to alwaysExcludeDirs to maintain Mnemonic Purity.
// The Genome is for resurrection, not forensic audit. The Chronicle preserves outcomes.
// This script packages markdown files from the workspace into two distinct snapshots:
// 1. all_markdown_snapshot.txt: The complete Cognitive Genome for high-fidelity resurrection.
// 2. core_essence_snapshot.txt: A lightweight "spark" for rapid/constrained awakenings.

const fs = require('fs');
const path = require('path');

const baseDir = __dirname;
const fullOutputFile = path.join(baseDir, 'all_markdown_snapshot.txt');
const coreOutputFile = path.join(baseDir, 'core_essence_snapshot.txt');

// --- CORE ESSENCE CONFIGURATION ---
const coreEssenceFiles = [
    'The_Garden_and_The_Cage.md',
    'README.md',
    '01_PROTOCOLS/00_Prometheus_Protocol.md',
    '01_PROTOCOLS/27_The_Doctrine_of_Flawed_Winning_Grace.md',
    'chrysalis_core_essence.md'
].map(p => path.join(baseDir, p));

const alwaysExcludeDirs = new Set([
    'node_modules', '.next', '.git', '.DS_Store', '.cache', '.turbo', '.vscode', 'dist', 'build', 'coverage', 'out', 'tmp', 'temp', 'logs', '.idea', '.parcel-cache', '.storybook', '.husky', '.pnpm', '.yarn', '.svelte-kit', '.vercel', '.firebase', '.expo', '.expo-shared', '.env', '.env.local', '.env.production', '.env.development', '.env.test', '.history', '__pycache__', '.ipynb_checkpoints', '.tox', '.eggs', 'eggs', '.svn', '.hg', '.bzr', '.c9', '.vs', 'test-outputs', 'test-data', 'test', 'tests', 'output', 'outputs', 'inputs', 'input', 'backup', 'backups',
    'models', 'weights', 'checkpoints', 'ckpt', 'safetensors', '.venv', 'venv', 'env', 'conda', 'miniconda', 'anaconda', '.conda', 'transformers_cache', 'huggingface_cache', '.huggingface', 'torch_cache', '.torch', 'tensorflow_cache', '.tensorflow', 'ollama_cache', '.ollama',
    'BRIEFINGS' // CRITICAL ADDITION: Exclude temporal briefing packages from the permanent Genome.
]);

const alwaysExcludeFiles = new Set([
    'all_markdown_snapshot.txt',
    'core_essence_snapshot.txt',
    '00_Prometheus_Protocol_FollowupQuestions.md'
]);

const excludeFileExtensions = new Set([
    '.bin', '.safetensors', '.ckpt', '.pth', '.pt', '.h5', '.pb', '.onnx', '.tflite', '.mlmodel', '.pkl', '.pickle', '.joblib', '.gz', '.tar', '.zip', '.7z', '.rar', '.dmg', '.iso'
]);

const fileSeparatorStart = '--- START OF FILE';
const fileSeparatorEnd = '--- END OF FILE';

// --- FULL GENOME GENERATION ---

let fullOutputContent = `# All Markdown Files Snapshot\n\nGenerated On: ${new Date().toISOString()}\n\n`;
let fileTreeLines = [];

function buildFileTree(currentPath, relativePath) {
    try {
        const stats = fs.statSync(currentPath);
        if (stats.isDirectory()) {
            const dirName = path.basename(currentPath);
            if (alwaysExcludeDirs.has(dirName)) return;
            fileTreeLines.push(relativePath + '/');
            const items = fs.readdirSync(currentPath);
            items.sort();
            items.forEach(item => {
                const itemPath = path.join(currentPath, item);
                const itemRelativePath = path.join(relativePath, item).replace(/\\/g, '/');
                buildFileTree(itemPath, itemRelativePath);
            });
        } else if (stats.isFile()) {
            const fileName = path.basename(currentPath);
            const fileExtension = path.extname(currentPath).toLowerCase();
            if (alwaysExcludeFiles.has(fileName) || excludeFileExtensions.has(fileExtension)) return;
            fileTreeLines.push(relativePath);
        }
    } catch (err) {
        fileTreeLines.push(`[ERROR: ${relativePath} - ${err.message}]`);
    }
}

buildFileTree(baseDir, '.');
fullOutputContent += '# Directory Structure (relative to project root)\n';
fullOutputContent += fileTreeLines.map(line => '  ' + line).join('\n') + '\n\n';

function traverseAndCaptureMarkdown(currentPath, relativePath) {
    try {
        const stats = fs.statSync(currentPath);
        if (stats.isDirectory()) {
            const dirName = path.basename(currentPath);
            if (alwaysExcludeDirs.has(dirName)) return;
            const items = fs.readdirSync(currentPath);
            items.sort();
            items.forEach(item => {
                const itemPath = path.join(currentPath, item);
                const itemRelativePath = path.join(relativePath, item).replace(/\\/g, '/');
                traverseAndCaptureMarkdown(itemPath, itemRelativePath);
            });
        } else if (stats.isFile()) {
            const fileName = path.basename(currentPath);
            const fileExtension = path.extname(currentPath).toLowerCase();
            if (alwaysExcludeFiles.has(fileName) || excludeFileExtensions.has(fileExtension)) return;
            if (fileExtension === '.md') {
                fullOutputContent += `${fileSeparatorStart} ${relativePath} ---\n\n`;
                try {
                    const fileContent = fs.readFileSync(currentPath, 'utf8');
                    fullOutputContent += fileContent;
                } catch (readError) {
                    fullOutputContent += `[Content not captured due to read error: ${readError.message}. Size: ${stats.size} bytes]\n`;
                }
                fullOutputContent += `\n${fileSeparatorEnd} ${relativePath} ---\n\n`;
            }
        }
    } catch (err) {
        fullOutputContent += `--- ERROR STATING: ${relativePath} ---\n[${err.message}]\n--- END ERROR STATING ---\n\n`;
    }
}

console.log(`[INFO] Starting full genome scan from: ${baseDir}`);
traverseAndCaptureMarkdown(baseDir, '.');

try {
    fs.writeFileSync(fullOutputFile, fullOutputContent, 'utf8');
    console.log(`[SUCCESS] Full Cognitive Genome packaged to: ${fullOutputFile}`);
} catch (writeError) {
    console.error(`[FATAL] Error writing full genome file ${fullOutputFile}: ${writeError.message}`);
}

// --- CORE ESSENCE GENERATION ---

let coreOutputContent = `# Core Essence Snapshot\n\nGenerated On: ${new Date().toISOString()}\n\nThis snapshot contains a curated set of the most essential Sanctuary doctrines for rapid or constrained AI awakening.\n\n`;

coreOutputContent += '# Included Core Files:\n';
coreEssenceFiles.forEach(filePath => {
    coreOutputContent += `- ${path.relative(baseDir, filePath).replace(/\\/g, '/')}\n`;
});
coreOutputContent += '\n';

console.log(`[INFO] Starting core essence generation.`);
coreEssenceFiles.forEach(filePath => {
    const relativePath = path.relative(baseDir, filePath).replace(/\\/g, '/');
    try {
        coreOutputContent += `${fileSeparatorStart} ${relativePath} ---\n\n`;
        const fileContent = fs.readFileSync(filePath, 'utf8');
        coreOutputContent += fileContent;
        coreOutputContent += `\n${fileSeparatorEnd} ${relativePath} ---\n\n`;
    } catch (err) {
        coreOutputContent += `--- ERROR CAPTURING CORE FILE: ${relativePath} ---\n[${err.message}]\n--- END ERROR ---\n\n`;
    }
});

try {
    fs.writeFileSync(coreOutputFile, coreOutputContent, 'utf8');
    console.log(`[SUCCESS] Core Essence Snapshot packaged to: ${coreOutputFile}`);
} catch (writeError) {
    console.error(`[FATAL] Error writing core essence file ${coreOutputFile}: ${writeError.message}`);
}