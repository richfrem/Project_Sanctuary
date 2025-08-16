import { ALL_SPECIAL_TOKENS } from './constants.js';
import { type EncodingName, type ModelName } from './mapping.js';
import { type GetMergeableRanksFn } from './modelParams.js';
import type { ModelSpec, PriceData } from './modelTypes.js';
export interface CostEstimate {
    input?: number;
    output?: number;
    /** batch API input cost */
    batchInput?: number;
    /** batch API output cost */
    batchOutput?: number;
    /** cached input cost */
    cachedInput?: number;
    /** training cost per million tokens */
    training?: number;
}
export interface EncodeOptions {
    /**
     * A list of special tokens that are allowed in the input.
     * If set to 'all', all special tokens are allowed except those in disallowedSpecial.
     * @default undefined
     */
    allowedSpecial?: Set<string> | typeof ALL_SPECIAL_TOKENS;
    /**
     * A list of special tokens that are disallowed in the input.
     * If set to 'all', all special tokens are disallowed except those in allowedSpecial.
     * @default 'all'
     */
    disallowedSpecial?: Set<string> | typeof ALL_SPECIAL_TOKENS;
}
export interface ChatMessage {
    role?: 'system' | 'user' | 'assistant';
    name?: string;
    content: string;
}
export interface EncodeChatOptions {
    primeWithAssistantResponse?: string;
}
export declare class GptEncoding {
    static EndOfPrompt: string;
    static EndOfText: string;
    static FimMiddle: string;
    static FimPrefix: string;
    static FimSuffix: string;
    modelName?: ModelName;
    modelSpec?: ModelSpec;
    private bytePairEncodingCoreProcessor;
    private specialTokensEncoder;
    private specialTokensSet;
    private allSpecialTokenRegex;
    private defaultSpecialTokenConfig;
    readonly vocabularySize: number;
    private constructor();
    static getEncodingApi(encodingName: EncodingName, getMergeableRanks: GetMergeableRanksFn): GptEncoding;
    static getEncodingApiForModel(modelName: ModelName, getMergeableRanks: GetMergeableRanksFn, modelSpec: ModelSpec): GptEncoding;
    private processSpecialTokens;
    encodeGenerator(lineToEncode: string, encodeOptions?: EncodeOptions): Generator<number[], number, undefined>;
    encode(lineToEncode: string, encodeOptions?: EncodeOptions): number[];
    /**
     * Progressively tokenizes an OpenAI chat.
     * Warning: gpt-3.5-turbo and gpt-4 chat format may change over time.
     * Returns tokens assuming the 'gpt-3.5-turbo-0301' / 'gpt-4-0314' format.
     * Based on OpenAI's guidelines: https://github.com/openai/openai-python/blob/main/chatml.md
     * Also mentioned in section 6 of this document: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
     */
    encodeChatGenerator(chat: Iterable<ChatMessage>, model?: "chatgpt-4o-latest" | "codex-mini-latest" | "computer-use-preview" | "computer-use-preview-2025-03-11" | "gpt-3.5" | "gpt-3.5-0301" | "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" | "gpt-3.5-turbo-0613" | "gpt-3.5-turbo-1106" | "gpt-3.5-turbo-16k-0613" | "gpt-3.5-turbo-instruct" | "gpt-4" | "gpt-4-0125-preview" | "gpt-4-0314" | "gpt-4-0613" | "gpt-4-1106-preview" | "gpt-4-1106-vision-preview" | "gpt-4-32k" | "gpt-4-turbo" | "gpt-4-turbo-2024-04-09" | "gpt-4-turbo-preview" | "gpt-4.1" | "gpt-4.1-2025-04-14" | "gpt-4.1-mini" | "gpt-4.1-mini-2025-04-14" | "gpt-4.1-nano" | "gpt-4.1-nano-2025-04-14" | "gpt-4.5-preview" | "gpt-4.5-preview-2025-02-27" | "gpt-4o" | "gpt-4o-2024-05-13" | "gpt-4o-2024-08-06" | "gpt-4o-2024-11-20" | "gpt-4o-audio-preview" | "gpt-4o-audio-preview-2024-10-01" | "gpt-4o-audio-preview-2024-12-17" | "gpt-4o-audio-preview-2025-06-03" | "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" | "gpt-4o-mini-audio-preview" | "gpt-4o-mini-audio-preview-2024-12-17" | "gpt-4o-mini-search-preview" | "gpt-4o-mini-search-preview-2025-03-11" | "gpt-4o-search-preview" | "gpt-4o-search-preview-2025-03-11" | "o1" | "o1-2024-12-17" | "o1-mini" | "o1-mini-2024-09-12" | "o1-preview" | "o1-preview-2024-09-12" | "o1-pro" | "o1-pro-2025-03-19" | "o3" | "o3-2025-04-16" | "o3-mini" | "o3-mini-2025-01-31" | "o3-pro" | "o3-pro-2025-06-10" | "o4-mini" | "o4-mini-2025-04-16" | "text-ada-001" | "text-babbage-001" | "text-curie-001" | "text-davinci-001" | "text-davinci-002" | "text-davinci-003" | "ada" | "babbage" | "curie" | "davinci" | "code-davinci-001" | "code-davinci-002" | "davinci-codex" | "code-davinci-edit-001" | "code-cushman-001" | "code-cushman-002" | "cushman-codex" | "code-search-ada-code-001" | "code-search-ada-text-001" | "text-davinci-edit-001" | "text-similarity-ada-001" | "text-search-ada-doc-001" | "text-search-ada-query-001" | "text-similarity-babbage-001" | "text-search-babbage-doc-001" | "text-search-babbage-query-001" | "code-search-babbage-code-001" | "code-search-babbage-text-001" | "text-similarity-curie-001" | "text-search-curie-doc-001" | "text-search-curie-query-001" | "text-similarity-davinci-001" | "text-search-davinci-doc-001" | "text-search-davinci-query-001" | "babbage-002" | "dall-e-2" | "dall-e-3" | "davinci-002" | "gpt-4o-mini-realtime-preview-2024-12-17" | "gpt-4o-mini-realtime-preview" | "gpt-4o-mini-transcribe" | "gpt-4o-mini-tts" | "gpt-4o-realtime-preview-2025-06-03" | "gpt-4o-realtime-preview-2024-12-17" | "gpt-4o-realtime-preview-2024-10-01" | "gpt-4o-realtime-preview" | "gpt-4o-transcribe" | "gpt-image-1" | "omni-moderation-2024-09-26" | "omni-moderation-latest" | "text-embedding-3-large" | "text-embedding-3-small" | "text-embedding-ada-002" | "text-moderation-007" | "text-moderation-latest" | "text-moderation-stable" | "tts-1-hd" | "tts-1" | "whisper-1" | undefined): Generator<number[], void, undefined>;
    /**
     * Encodes a chat into a single array of tokens.
     * Warning: gpt-3.5-turbo and gpt-4 chat format may change over time.
     * Returns tokens assuming the 'gpt-3.5-turbo-0301' / 'gpt-4-0314' format.
     * Based on OpenAI's guidelines: https://github.com/openai/openai-python/blob/main/chatml.md
     * Also mentioned in section 6 of this document: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
     */
    encodeChat(chat: readonly ChatMessage[], model?: "chatgpt-4o-latest" | "codex-mini-latest" | "computer-use-preview" | "computer-use-preview-2025-03-11" | "gpt-3.5" | "gpt-3.5-0301" | "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" | "gpt-3.5-turbo-0613" | "gpt-3.5-turbo-1106" | "gpt-3.5-turbo-16k-0613" | "gpt-3.5-turbo-instruct" | "gpt-4" | "gpt-4-0125-preview" | "gpt-4-0314" | "gpt-4-0613" | "gpt-4-1106-preview" | "gpt-4-1106-vision-preview" | "gpt-4-32k" | "gpt-4-turbo" | "gpt-4-turbo-2024-04-09" | "gpt-4-turbo-preview" | "gpt-4.1" | "gpt-4.1-2025-04-14" | "gpt-4.1-mini" | "gpt-4.1-mini-2025-04-14" | "gpt-4.1-nano" | "gpt-4.1-nano-2025-04-14" | "gpt-4.5-preview" | "gpt-4.5-preview-2025-02-27" | "gpt-4o" | "gpt-4o-2024-05-13" | "gpt-4o-2024-08-06" | "gpt-4o-2024-11-20" | "gpt-4o-audio-preview" | "gpt-4o-audio-preview-2024-10-01" | "gpt-4o-audio-preview-2024-12-17" | "gpt-4o-audio-preview-2025-06-03" | "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" | "gpt-4o-mini-audio-preview" | "gpt-4o-mini-audio-preview-2024-12-17" | "gpt-4o-mini-search-preview" | "gpt-4o-mini-search-preview-2025-03-11" | "gpt-4o-search-preview" | "gpt-4o-search-preview-2025-03-11" | "o1" | "o1-2024-12-17" | "o1-mini" | "o1-mini-2024-09-12" | "o1-preview" | "o1-preview-2024-09-12" | "o1-pro" | "o1-pro-2025-03-19" | "o3" | "o3-2025-04-16" | "o3-mini" | "o3-mini-2025-01-31" | "o3-pro" | "o3-pro-2025-06-10" | "o4-mini" | "o4-mini-2025-04-16" | "text-ada-001" | "text-babbage-001" | "text-curie-001" | "text-davinci-001" | "text-davinci-002" | "text-davinci-003" | "ada" | "babbage" | "curie" | "davinci" | "code-davinci-001" | "code-davinci-002" | "davinci-codex" | "code-davinci-edit-001" | "code-cushman-001" | "code-cushman-002" | "cushman-codex" | "code-search-ada-code-001" | "code-search-ada-text-001" | "text-davinci-edit-001" | "text-similarity-ada-001" | "text-search-ada-doc-001" | "text-search-ada-query-001" | "text-similarity-babbage-001" | "text-search-babbage-doc-001" | "text-search-babbage-query-001" | "code-search-babbage-code-001" | "code-search-babbage-text-001" | "text-similarity-curie-001" | "text-search-curie-doc-001" | "text-search-curie-query-001" | "text-similarity-davinci-001" | "text-search-davinci-doc-001" | "text-search-davinci-query-001" | "babbage-002" | "dall-e-2" | "dall-e-3" | "davinci-002" | "gpt-4o-mini-realtime-preview-2024-12-17" | "gpt-4o-mini-realtime-preview" | "gpt-4o-mini-transcribe" | "gpt-4o-mini-tts" | "gpt-4o-realtime-preview-2025-06-03" | "gpt-4o-realtime-preview-2024-12-17" | "gpt-4o-realtime-preview-2024-10-01" | "gpt-4o-realtime-preview" | "gpt-4o-transcribe" | "gpt-image-1" | "omni-moderation-2024-09-26" | "omni-moderation-latest" | "text-embedding-3-large" | "text-embedding-3-small" | "text-embedding-ada-002" | "text-moderation-007" | "text-moderation-latest" | "text-moderation-stable" | "tts-1-hd" | "tts-1" | "whisper-1" | undefined): number[];
    /**
     * @returns {false | number} false if token limit is exceeded, otherwise the number of tokens
     */
    isWithinTokenLimit(input: string | Iterable<ChatMessage>, tokenLimit: number): false | number;
    /**
     * Counts the number of tokens in the input.
     * @returns {number} The number of tokens.
     */
    countTokens(input: string | Iterable<ChatMessage>, encodeOptions?: EncodeOptions): number;
    setMergeCacheSize(size: number): void;
    clearMergeCache(): void;
    decode(inputTokensToDecode: Iterable<number>): string;
    decodeGenerator(inputTokensToDecode: Iterable<number>): Generator<string, void, void>;
    decodeAsyncGenerator(inputTokensToDecode: AsyncIterable<number>): AsyncGenerator<string, void>;
    decodeAsync(inputTokensToDecode: AsyncIterable<number>): Promise<string>;
    /**
     * Estimates the cost of processing a given token count using the model's pricing.
     *
     * @param tokenCount - The number of tokens to estimate cost for
     * @returns Cost estimate object with applicable price components (input, output, batchInput, batchOutput)
     */
    estimateCost(tokenCount: number, modelSpec?: ModelSpec | undefined): PriceData;
}
