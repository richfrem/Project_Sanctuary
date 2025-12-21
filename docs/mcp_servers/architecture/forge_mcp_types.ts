/**
 * Forge MCP Server - Type Definitions
 * Model Lifecycle Orchestrator
 * Version: 1.0
 */

// ============================================================================
// Forge Configuration Types
// ============================================================================

export interface ForgeConfig {
    forge_id: string;                    // Unique ID for idempotency
    authorization_task_id: number;       // Link to Task MCP entry
    hyperparameters: ForgeHyperparameters;
}

export interface ForgeHyperparameters {
    base_model: string;                  // e.g., "Qwen/Qwen2-7B-Instruct"
    dataset_path: string;                // Path to training data
    lora_rank: number;                   // LoRA rank (e.g., 64)
    lora_alpha: number;                  // LoRA alpha (e.g., 16)
    max_steps: number;                   // Training steps
    learning_rate: number;               // Learning rate
    batch_size: number;                  // Batch size
    gradient_accumulation_steps: number; // Gradient accumulation
    warmup_steps: number;                // Warmup steps
    save_steps: number;                  // Checkpoint frequency
    logging_steps: number;               // Logging frequency
}

// ============================================================================
// Job Management Types
// ============================================================================

export interface ForgeJobResult {
    job_id: string;
    status: "queued" | "running" | "completed" | "failed";
    start_time: string;
    hyperparameters: ForgeHyperparameters;
}

export interface JobStatus {
    status: "queued" | "running" | "completed" | "failed";
    progress: number;                    // 0-100
    logs_snippet: string;                // Last 500 chars of logs
    elapsed_seconds: number;
    current_step: string;                // e.g., "fine_tuning", "merging_adapter"
    estimated_completion?: string;       // ISO timestamp
}

export interface JobProgress {
    step: number;
    total_steps: number;
    loss: number;
    learning_rate: number;
    samples_per_second: number;
}

// ============================================================================
// Artifact Management Types
// ============================================================================

export interface ArtifactPackage {
    gguf_path: string;
    modelfile_path: string;
    ollama_model_name: string;
    verification_status: "AWAITING_TESTS" | "PASSED" | "FAILED";
    sha256_hash: string;                 // P101-style integrity
    quantization: "Q4_K_M" | "Q8_0" | "F16";
}

export interface InferenceTestResult {
    all_passed: boolean;
    test_results: Array<{
        prompt: string;
        response: string;
        latency_ms: number;
        quality_score: number;            // 0-1
        passed: boolean;
    }>;
    avg_latency_ms: number;
    quality_score: string;              // "excellent" | "good" | "poor"
}

export interface PublishResult {
    url: string;                        // Hugging Face repo URL
    commit_hash: string;
    uploaded_files: string[];
    upload_time_seconds: number;
}

export interface ArtifactDownload {
    local_path: string;
    repo_name: string;
    files_downloaded: string[];
    verification_status: "verified" | "failed";
}

// ============================================================================
// Resource Management Types
// ============================================================================

export interface ResourceStatus {
    cuda_available: boolean;
    gpu_memory_gb: number;
    gpu_memory_free_gb: number;
    disk_space_gb: number;
    ml_env_active: boolean;
    missing_dependencies: string[];
    can_start_job: boolean;
    blocking_reason?: string;
}

export interface CUDAEnvironment {
    check_gpu_availability(): boolean;
    get_gpu_memory(): { total: number; free: number };
    activate_ml_env(): boolean;
    verify_dependencies(): string[];    // Returns missing deps
}

// ============================================================================
// Script Execution Types
// ============================================================================

export interface ScriptWhitelist {
    "forge_whole_genome_dataset.py": boolean;
    "fine_tune.py": boolean;
    "merge_adapter.py": boolean;
    "inference.py": boolean;
    "convert_to_gguf.py": boolean;
    "create_modelfile.py": boolean;
    "upload_to_huggingface.py": boolean;
}

export interface ScriptResult {
    script_name: string;
    exit_code: number;
    stdout: string;
    stderr: string;
    execution_time_seconds: number;
}

// ============================================================================
// Forge MCP Tool Signatures
// ============================================================================

export interface ForgeMCP {
    /**
     * Initiate full model forge pipeline (async)
     * Steps: Create dataset → Fine-tune → Merge adapter
     */
    initiate_model_forge(config: ForgeConfig): Promise<ForgeJobResult>;

    /**
     * Get status of running forge job
     */
    get_forge_job_status(job_id: string): Promise<JobStatus>;

    /**
     * Package completed model into deployment artifacts
     * Steps: Convert GGUF → Create Modelfile → Import Ollama
     */
    package_and_deploy_artifact(
        job_id: string,
        quantization: "Q4_K_M" | "Q8_0" | "F16"
    ): Promise<ArtifactPackage>;

    /**
     * Run automated inference tests on model
     */
    run_inference_test(
        model_path: string,
        prompts: string[]
    ): Promise<InferenceTestResult>;

    /**
     * Publish artifact to Hugging Face
     */
    publish_to_registry(
        job_id: string,
        repo_name: string,
        commit_message?: string
    ): Promise<PublishResult>;

    /**
     * Download artifact from Hugging Face
     */
    retrieve_registry_artifact(
        repo_name: string,
        local_path?: string
    ): Promise<ArtifactDownload>;

    /**
     * Check if system has resources to start forge job
     */
    check_resource_availability(): Promise<ResourceStatus>;
}

// ============================================================================
// Safety Validation Types
// ============================================================================

export interface ForgeValidationResult extends ValidationResult {
    cuda_check_passed: boolean;
    resource_check_passed: boolean;
    script_whitelist_passed: boolean;
    task_linkage_verified: boolean;
}

export interface ForgeSafetyRules {
    // Environment gate
    require_cuda_marker: boolean;       // CUDA_FORGE_ACTIVE must be set

    // Resource checks
    min_gpu_memory_gb: number;          // Minimum GPU memory required
    min_disk_space_gb: number;          // Minimum disk space required

    // Task linkage
    require_task_authorization: boolean; // Must link to Task MCP entry

    // Script whitelist
    allowed_scripts: ScriptWhitelist;

    // Artifact integrity
    require_sha256_validation: boolean;  // P101-style integrity check
}
