/**
 * Shared Infrastructure Type Definitions
 * Project Sanctuary MCP Ecosystem
 * Version: 1.1 (Refined based on feedback)
 */

// ============================================================================
// Validation Result Types
// ============================================================================

/**
 * Standard validation result returned by all validators
 */
export interface ValidationResult {
  is_valid: boolean;
  errors?: ValidationError[];
  warnings?: ValidationWarning[];
}

export interface ValidationError {
  field: string;
  message: string;
  severity: "error";
}

export interface ValidationWarning {
  field: string;
  message: string;
  severity: "warning";
}

// ============================================================================
// Safety Validator
// ============================================================================

export enum ProtectionLevel {
  UNRESTRICTED = "unrestricted",           // No restrictions
  WRITE_WITH_VALIDATION = "write_with_validation",  // Standard validation
  WRITE_WITH_APPROVAL = "write_with_approval",      // Requires approval
  READ_ONLY = "read_only",                 // Cannot modify
  FORBIDDEN = "forbidden"                  // Cannot access
}

export enum RiskLevel {
  SAFE = "safe",           // No risk, auto-execute
  MODERATE = "moderate",   // Some risk, validation required
  DANGEROUS = "dangerous"  // High risk, blocked or requires approval
}

export interface RiskAssessment {
  risk_level: RiskLevel;
  allowed: boolean;
  reason?: string;
  requires_approval?: boolean;
  approval_id?: string;
}

export interface SafetyValidator {
  /**
   * Validate file path against project boundaries and protected paths
   */
  validate_path(path: string): ValidationResult;
  
  /**
   * Check if file is protected (cannot be modified without approval)
   */
  is_protected_file(path: string): boolean;
  
  /**
   * Get protection level for a specific path
   * Based on .agent/git_safety_rules.md
   */
  get_protection_level(path: string): ProtectionLevel;
  
  /**
   * Assess risk level of an operation
   */
  assess_risk(operation: string, params: Record<string, any>): RiskAssessment;
  
  /**
   * Validate commit message format (conventional commits)
   */
  validate_commit_message(message: string): ValidationResult;
  
  /**
   * Check if operation requires user approval
   */
  requires_approval(operation: string, params: Record<string, any>): boolean;
}

// ============================================================================
// Schema Validator
// ============================================================================

export interface ChronicleEntry {
  entry_number: number;      // Auto-generated, sequential
  title: string;
  date: string;              // ISO format
  author: string;            // e.g., "GUARDIAN-02"
  content: string;           // Markdown
  status?: string;           // e.g., "CANONICAL", "DRAFT"
  classification?: string;   // e.g., "STRATEGIC"
}

export interface Protocol {
  number: number;            // Unique
  title: string;
  classification: string;    // e.g., "Foundational"
  content: string;           // Markdown
  status: string;            // e.g., "Canonical", "Draft"
  version: string;           // e.g., "v2.0"
  linked_protocols?: number[];
}

export interface ADR {
  number: number;            // Sequential
  title: string;
  date: string;              // ISO format
  status: string;            // "Proposed", "Accepted", "Superseded"
  context: string;
  decision: string;
  consequences: string;
  supersedes?: number[];
}

export interface Task {
  number: number;            // Unique
  title: string;
  description: string;       // Markdown
  status: string;            // "Backlog", "Active", "Completed"
  priority: string;          // "High", "Medium", "Low"
  estimated_effort?: string;
  dependencies?: number[];
}

export interface SchemaValidator {
  /**
   * Validate chronicle entry schema
   */
  validate_chronicle_entry(entry: Partial<ChronicleEntry>): ValidationResult;
  
  /**
   * Validate protocol schema
   * Enforces version bump for canonical protocol updates
   */
  validate_protocol(protocol: Partial<Protocol>, is_update?: boolean, current_version?: string): ValidationResult;
  
  /**
   * Validate ADR schema
   */
  validate_adr(adr: Partial<ADR>): ValidationResult;
  
  /**
   * Validate task schema
   * Includes circular dependency detection
   */
  validate_task(task: Partial<Task>, all_tasks?: Task[]): ValidationResult;
  
  /**
   * Detect circular dependencies in task graph
   */
  detect_circular_dependencies(task_id: number, dependencies: number[], all_tasks: Task[]): boolean;
  
  /**
   * Validate status transition (for tasks/ADRs)
   */
  validate_status_transition(current_status: string, new_status: string, entity_type: "task" | "adr"): ValidationResult;
}

// ============================================================================
// Git Operations
// ============================================================================

export interface CommitManifest {
  guardian_approval: string;
  approval_timestamp: string;
  commit_message: string;
  files: Array<{
    path: string;
    sha256: string;
  }>;
}

export interface CommitResult {
  commit_hash: string;
  manifest_path: string;
  files_committed: string[];
}

export interface GitOperations {
  /**
   * Generate commit manifest with SHA-256 hashes
   */
  generate_manifest(files: string[]): CommitManifest;
  
  /**
   * Commit with Protocol 101 compliance
   */
  commit_with_manifest(
    files: string[],
    message: string,
    push?: boolean
  ): Promise<CommitResult>;
  
  /**
   * Validate commit message format
   */
  validate_commit_message(message: string): ValidationResult;
}

// ============================================================================
// MCP Tool Response Types
// ============================================================================

export interface MCPToolResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}

export interface FileOperationResult {
  file_path: string;
  commit_hash?: string;
  manifest_path?: string;
}

export interface QueryResult<T> {
  results: T[];
  total_count: number;
  query_time_ms: number;
}
