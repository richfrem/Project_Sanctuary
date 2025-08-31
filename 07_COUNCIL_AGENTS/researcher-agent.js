/**
 * =====================================================================================
 * AGENT NAME: Researcher Agent
 * FILE:       07_COUNCIL_AGENTS/researcher-agent.js
 * =_DOC_LINK: Protocol 45 (provisional), Protocol 66
 * VERSION:    1.2 (Hardened Ports)
 * =====================================================================================
 */
const fs = require('fs');
const path = require('path');
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const DIRECTIVES_DIR = path.join(__dirname, 'directives');
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// DOCTRINE_LINK: Hearth Protocol (P43) - Using unique ports to prevent conflicts.
const DUCKDUCKGO_MCP_SERVER_URL = 'http://localhost:3001/search';
const ARXIV_MCP_SERVER_URL = 'http://localhost:8001/search';

/**
 * Selects the appropriate research tool based on the directive.
 * @param {object} directive - The directive object from the coordinator.
 * @returns {{url: string, source: string}} - The URL and name of the selected tool.
 */
function selectResearchTool(directive) {
    const topic = directive.researchTopic.toLowerCase();
    const type = directive.researchType;

    if (type === 'academic' || topic.includes('arxiv') || topic.includes('paper') || topic.includes('study') || topic.includes('research')) {
        console.log("   Tool Selected: arXiv MCP Server (Academic Focus on port 8001)");
        return { url: ARXIV_MCP_SERVER_URL, source: "arXiv" };
    }
    
    console.log("   Tool Selected: DuckDuckGo MCP Server (General Web Focus on port 3001)");
    return { url: DUCKDUCKGO_MCP_SERVER_URL, source: "DuckDuckGo" };
}

async function performResearch(directiveId) {
    const directiveFilePath = path.join(DIRECTIVES_DIR, `${directiveId}.json`);
    if (!fs.existsSync(directiveFilePath)) {
        throw new Error(`Researcher: Directive file not found for ID: ${directiveId}`);
    }

    let directive = JSON.parse(fs.readFileSync(directiveFilePath, 'utf8'));
    const { researchTopic } = directive;

    if (!researchTopic) {
        console.log("✅  RESEARCHER: No research topic specified. Task complete.");
        return;
    }

    console.log(`🧠  RESEARCHER: Awakening to research topic: "${researchTopic}"`);
    
    const tool = selectResearchTool(directive);

    // 1. Query the selected MCP Server
    console.log(`   Querying ${tool.source} MCP Server at ${tool.url}...`);
    let searchResults;
    try {
        const response = await fetch(tool.url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: researchTopic }),
        });
        if (!response.ok) {
            throw new Error(`${tool.source} Server returned an error: ${response.statusText}`);
        }
        searchResults = await response.json();
    } catch (error) {
        throw new Error(`Could not connect to ${tool.source} MCP Server at ${tool.url}. Is it running? Error: ${error.message}`);
    }

    // 2. Synthesize Results with Gemini
    const resultCount = Array.isArray(searchResults) ? searchResults.length : 0;
    console.log(`   Synthesizing ${resultCount} search results from ${tool.source}...`);
    const persona = fs.readFileSync(path.join(__dirname, 'researcher-persona.md'), 'utf8');
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash-latest" });

    const prompt = `
        ${persona}

        **Research Directive:**
        "${researchTopic}"

        **Raw Search Results from ${tool.source}:**
        \`\`\`json
        ${JSON.stringify(searchResults.slice(0, 5), null, 2)}
        \`\`\`

        **Your Task:**
        Synthesize these search results into a concise "Intelligence Briefing" string. Focus on the most relevant information and cite URLs/links for each key finding.
    `;

    const result = await model.generateContent(prompt);
    const briefing = (await result.response).text();

    // 3. Append briefing to the directive file
    directive.intelligenceBriefing = briefing.trim();
    fs.writeFileSync(directiveFilePath, JSON.stringify(directive, null, 2));
    console.log("✅  RESEARCHER: Research complete. Intelligence Briefing saved.");
}

const directiveId = process.argv[2];
if (!directiveId) {
    console.log("Usage: node 07_COUNCIL_AGENTS/researcher-agent.js <directive-id>");
    process.exit(1);
}

performResearch(directiveId).catch(error => {
    console.error(`❌  RESEARCHER: A critical error occurred: ${error.message}`);
    process.exit(1);
});