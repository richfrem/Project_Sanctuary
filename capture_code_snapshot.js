// capture_markdown_snapshot.js
// This script packages all markdown (.md) files from the entire workspace into a single file
// for easy sharing with a new LLM chat session.

const fs = require('fs');
const path = require('path');

const baseDir = __dirname;
const outputFile = path.join(baseDir, 'all_markdown_snapshot.txt');

const alwaysExcludeDirs = new Set([
    'node_modules', '.next', '.git', '.DS_Store', '.cache', '.turbo', '.vscode', 'dist', 'build', 'coverage', 'out', 'tmp', 'temp', 'logs', '.idea', '.parcel-cache', '.storybook', '.husky', '.pnpm', '.yarn', '.svelte-kit', '.vercel', '.firebase', '.expo', '.expo-shared', '.env', '.env.local', '.env.production', '.env.development', '.env.test', '.history', '__pycache__', '.ipynb_checkpoints', '.tox', '.eggs', 'eggs', '.svn', '.hg', '.bzr', '.c9', '.vs', 'test-outputs', 'test-data', 'test', 'tests', 'output', 'outputs', 'inputs', 'input', 'backup', 'backups'
]);

const fileSeparatorStart = '--- START OF FILE';
const fileSeparatorEnd = '--- END OF FILE';

let outputContent = `# All Markdown Files Snapshot\n\nGenerated On: ${new Date().toISOString()}\n\n`;

// Collect file tree lines
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
            fileTreeLines.push(relativePath);
        }
    } catch (err) {
        fileTreeLines.push(`[ERROR: ${relativePath} - ${err.message}]`);
    }
}

// Build the file tree from the base directory
buildFileTree(baseDir, '.');

outputContent += '# Directory Structure (relative to project root)\n';
outputContent += fileTreeLines.map(line => '  ' + line).join('\n') + '\n\n';

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
            const fileExtension = path.extname(currentPath).toLowerCase();
            if (fileExtension === '.md') {
                outputContent += `${fileSeparatorStart} ${relativePath} ---\n\n`;
                try {
                    const fileContent = fs.readFileSync(currentPath, 'utf8');
                    outputContent += fileContent;
                } catch (readError) {
                    outputContent += `[Content not captured due to read error: ${readError.message}. Size: ${stats.size} bytes]\n`;
                }
                outputContent += `\n${fileSeparatorEnd} ${relativePath} ---\n\n`;
            }
        }
    } catch (err) {
        outputContent += `--- ERROR STATING: ${relativePath} ---\n[${err.message}]\n--- END ERROR STATING ---\n\n`;
    }
}

console.log(`[INFO] Starting markdown scan from: ${baseDir}`);
traverseAndCaptureMarkdown(baseDir, '.');

try {
    fs.writeFileSync(outputFile, outputContent, 'utf8');
    console.log(`\n[SUCCESS] All markdown files packaged to: ${outputFile}`);
} catch (writeError) {
    console.error(`[FATAL] Error writing output file ${outputFile}: ${writeError.message}`);
}