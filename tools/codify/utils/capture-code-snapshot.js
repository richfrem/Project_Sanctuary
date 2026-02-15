#!/usr/bin/env node
/**
 * capture-code-snapshot.js (CLI)
 * =====================================
 *
 * Purpose:
 *     Generates a single text file snapshot of code files for LLM context sharing.
 *
 * Layer: Curate / Documentation
 *
 * Usage Examples:
 *     node tools/codify/utils/capture-code-snapshot.js --help
 *
 * CLI Arguments:
 *     (None detected)
 *
 * Key Functions:
 *     - collectFiles()
 *
 * Consumed by:
 *     (Unknown)
 */

const fs = require('fs');
const path = require('path');
const ignore = require('ignore');

// --- Config ---
const exts = ['.md', '.txt', '.tf', '.json'];
const outputFile = 'llm_code_snapshot.txt';
const gitignoreFile = '.gitignore';
const rootDir = process.cwd();

// --- Load .gitignore rules ---
let ig = ignore();
const gitignorePath = path.join(rootDir, gitignoreFile);
if (fs.existsSync(gitignorePath)) {
  const gitignoreContent = fs.readFileSync(gitignorePath, 'utf8');
  ig = ig.add(gitignoreContent);
}

// --- Additional exclusions (security and clutter) ---
ig.add([
  // Package managers and dependencies
  'node_modules/',
  '.pnp/',
  '.yarn/',
  'packages/',

  // Build outputs
  'bin/',
  'obj/',
  'dist/',
  'build/',
  '.next/',
  'out/',

  // Test outputs
  'test-output/',
  'test-outputs/',
  'coverage/',

  // IDE and editor files
  '.git/',
  '.vscode/',
  '.vs/',
  '.idea/',
  '*.suo',
  '*.user',
  '*.userosscache',
  '*.sln.docstates',

  // OS files
  '.DS_Store',
  'Thumbs.db',
  'desktop.ini',

  // Environment and secrets
  '*.env',
  '*.env.local',
  '*.env.development.local',
  '*.env.test.local',
  '*.env.production.local',
  'azure.env',
  'azure.env.old',
  'secrets/',
  '.env.example',

  // Archives and backups
  'ARCHIVE/',
  'archive/',
  'backup/',
  '*.bak',
  '*.old',
  '*-backup/',

  // Logs and temporary files
  '*.log',
  'logs/',
  '*.tmp',
  '*.temp',
  '*.swp',
  '*~',

  // Large generated files
  'package-lock.json',
  'yarn.lock',
  'pnpm-lock.yaml',
  '*.min.js',
  '*.min.css',

  // Project-specific exclusions
  'OracleFormsSourceFiles/XML/',
  'OracleFormsSourceFiles/pll/',
  'OracleFormsSourceFiles/Reports/',
  'LLMConversionAttempts/LLM-Attempt1/',
  'LLMConversionAttempts/LLM-Attempt2/',
  'LLMConversionAttempts/LLM-Attempt3/',
  'LLMConversionAttempts/LLM-Attempt4/',
  'OracleSQL/exportedSQLFromSVN/',
  'OracleSQL/Packages/',
  'test-data/',
  'test-harness/',

  // Output file itself
  'llm_code_snapshot.txt',
  'code_snapshot.txt'
]);

// --- Recursively collect files ---
function collectFiles(dir) {
  let files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    const relPath = path.relative(rootDir, fullPath);
    if (ig.ignores(relPath)) continue;
    if (entry.isDirectory()) {
      files = files.concat(collectFiles(fullPath));
    } else if (exts.includes(path.extname(entry.name).toLowerCase())) {
      files.push(relPath);
    }
  }
  return files;
}

// --- Main ---
const allFiles = collectFiles(rootDir);
let output = `# LLM Code Snapshot\nGenerated: ${new Date().toISOString()}\n\n`;

for (const file of allFiles) {
  output += `--- START OF FILE: ${file} ---\n`;
  try {
    output += fs.readFileSync(path.join(rootDir, file), 'utf8');
  } catch (e) {
    output += `[Error reading file: ${e.message}]\n`;
  }
  output += `\n--- END OF FILE: ${file} ---\n\n`;
}

fs.writeFileSync(outputFile, output, 'utf8');
console.log(`Snapshot written to ${outputFile}`);