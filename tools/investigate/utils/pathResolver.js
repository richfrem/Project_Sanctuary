/**
 * Path Resolver Utility
 * =====================
 * Standardizes path resolution relative to the PROJECT_ROOT.
 * Handles cross-platform differences (Windows/WSL) and ensures API security.
 * 
 * ADR Reference: docs/ADRs/017-standardize-path-resolution.md
 */

import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Determine Project Root (try env, fallback to relative calculation)
// We are in tools/investigate/utils/pathResolver.js
// So root is 3 levels up: ../../../
const DETECTED_ROOT = path.resolve(__dirname, '../../../');
const PROJECT_ROOT = process.env.PROJECT_ROOT || DETECTED_ROOT;

console.log(`[PathResolver] Project Root set to: ${PROJECT_ROOT}`);

/**
 * Normalizes any path to use forward slashes (Unix style).
 * Useful for consistency in API responses.
 */
function normalizeSeparators(p) {
    return p.replace(/\\/g, '/');
}

/**
 * Converts an absolute system path to a project-relative path.
 * Handles mixed slashes and WSL /mnt/ prefixes if running in WSL but accessing Windows paths.
 */
function toRelative(absolutePath) {
    if (!absolutePath) return '';

    // Normalize inputs
    let normalizedAbs = normalizeSeparators(absolutePath);
    let normalizedRoot = normalizeSeparators(PROJECT_ROOT);

    // Handle WSL /mnt/c vs C:/ mismatch
    // If we are in WSL but query returned C:/ (or vice versa), we attempt to strip standard prefixes

    // 1. If path starts with Root, strip it
    if (normalizedAbs.startsWith(normalizedRoot)) {
        let rel = normalizedAbs.slice(normalizedRoot.length);
        if (rel.startsWith('/')) rel = rel.slice(1);
        return rel;
    }

    // 2. Fallback: Heuristic Check for common known folders
    // If path is like ".../legacy-system/...", we grab from legacy-system onwards
    const landmarks = ['/legacy-system/', '/docs/', '/tools/'];
    for (const mark of landmarks) {
        const idx = normalizedAbs.indexOf(mark);
        if (idx !== -1) {
            // Return "legacy-system/..." (stripping leading slash)
            return normalizedAbs.slice(idx + 1);
        }
    }

    return normalizedAbs; // Return original if no relative match found (safe fallback)
}

/**
 * Converts a project-relative path to a safe absolute system path.
 * Prevents traversal outside project root.
 */
function toAbsolute(relativePath) {
    if (!relativePath) return null;

    // Prevent traversal
    if (relativePath.includes('..')) {
        console.warn(`[PathResolver] Blocked traversal attempt: ${relativePath}`);
        return null;
    }

    return path.resolve(PROJECT_ROOT, relativePath);
}

/**
 * Resolves a path from an environment variable (or default) relative to PROJECT_ROOT.
 * Standardizes configuration loading.
 */
function resolveConfigPath(envVar, defaultRelPath) {
    const relPath = process.env[envVar] || defaultRelPath;
    if (!relPath) return null;
    return toAbsolute(relPath);
}

export {
    PROJECT_ROOT,
    toRelative,
    toAbsolute,
    resolveConfigPath,
    normalizeSeparators
};
