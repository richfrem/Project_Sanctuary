# Architecture & Capability Matrix: Autonomous Agent Platforms

## 1. High-Level Taxonomy & System Roles

| System | Primary Environment | Core Differentiators | Best Delegation Target | Integration Contract |
| :--- | :--- | :--- | :--- | :--- |
| **xAI: Grok Bot** | Cloud VM + iOS Frameworks | Autonomous PSTN telephony, native Apple Pay/Link checkout, disposable `@mail.grokbot.com` inbox | Real-world consumer tasks, automated merchant purchases, phone calls, mobile reminders/home control | Asynchronous Git handoff or webhook (No stdio REPL) |
| **Anthropic: Claude Duet** | Enterprise Cloud Connectors | Proactive outreach (initiates messages), human-in-the-loop business triage | Morning briefing, communications channel risk audit, accounting/CRM follow-up | Upstream task planner & brief generator |
| **OpenAI: Dots (Operator)** | Sandboxed Desktop / Web VM | OS-level GUI computer use (pixel vision, mouse/keyboard, raw browser interaction) | Legacy web scraping, multi-step web forms lacking public APIs, headless browser pipelines | Headless GUI workhorse / automation runner |
| **Google: Gemini Agent** | Chrome Engine + Google Workspace | Native enterprise coworker identity, real-time screen/video sight (Project Astra), Deep Research | Cross-tab web navigation, multi-source technical research dossiers, Docs/Sheets automation | Enterprise knowledge layer & research synthesizer |

---

## 2. Platform Deep-Dives for Multi-Agent Orchestration

### A. Grok Bot (xAI)
* **Execution Boundary**: Isolated remote cloud VM paired with device-level iOS framework hooks.
* **Identity & Auth**: Dedicated address (`@mail.grokbot.com`) for independent service signups. Integrated with 1Password, Apple Pay, and Stripe Link for one-time virtual card checkouts.
* **Core Strengths**:
  * Outbound PSTN phone calling with automated hold handling and structured note summarization.
  * Native iPhone API control (Apple Home, Reminders, Calendar, alarms, location).
  * In-chat interactive "Pages" (lightweight, shareable web dashboards).
* **CLI/Agent Integration Pattern**: Asynchronous webhook or Git handoff. No local stdio pipe. Best leveraged for external operational tasks via AgentMail triggers or committed markdown briefs.

### B. Claude Duet (Anthropic)
* **Execution Boundary**: Multi-SaaS enterprise connector framework operating under strict safety boundaries.
* **Identity & Auth**: Authenticated enterprise workspace identity (Google Workspace, Microsoft 365, Slack, Salesforce, Gusto, QuickBooks).
* **Core Strengths**:
  * Proactive initiation: Audits communication channels in the background and pings the user first with prioritized risk flags and drafts.
  * Human-in-the-loop governance: Stages sensitive actions (payments, outbound emails) behind explicit one-click confirmation gates.
  * Executive dashboard generation via "Make" tab (financial models, live health trackers).
* **CLI/Agent Integration Pattern**: Upstream task planner. Identifies operational blockers and assigns code or technical remediation tickets to developer CLI loops.

### C. Dots / Operator (OpenAI)
* **Execution Boundary**: Sandboxed virtual machine running full desktop and browser viewports.
* **Identity & Auth**: ChatGPT enterprise identity, headless session profiles, native OpenAI checkout rails.
* **Core Strengths**:
  * Direct vision-action computer use: Interprets GUI pixel streams to click buttons, scroll, copy-paste across desktop apps, and drive legacy web tools that lack APIs.
  * Deep autonomous loops: Multi-hour asynchronous task execution with self-correcting error recovery.
  * Integrated sandboxed code execution for tabular data manipulation and file outputs.
* **CLI/Agent Integration Pattern**: Headless browser workhorse. Handles visual web tasks, complex web crawling, and form submission flows that CLI `curl`/`fetch` scripts cannot traverse.

### D. Gemini Agent (Google)
* **Execution Boundary**: Native Chrome browser runtime + deep Google Workspace integration.
* **Identity & Auth**: First-class corporate coworker identity (dedicated `@company.com` account, internal Drive partitions, Slack/Chat presence, Doc `@mentions`).
* **Core Strengths**:
  * Chrome-native visual navigation: Operates across multiple active browser tabs with cross-referencing against Gmail and Drive.
  * Deep Research engine: Conducts 5 to 10 minute autonomous web sweeps, cross-referencing dozens of primary technical sources to produce structured Google Docs.
  * Real-time multimodal streaming (Project Astra): Live audio/video screen and physical camera ingestion for active environment debugging.
* **CLI/Agent Integration Pattern**: Enterprise knowledge layer. Synthesizes broad documentation sweeps and updates team spreadsheets/docs, providing clean contextual references for CLI coding engines (`agy`, Claude Code).

---

## 3. Commercial Subscription Tiers & Bundling

* **Cursor Premium ($30/mo)**:
  * Includes **Grok Bot** natively alongside Cursor IDE, cloud agents & automations, and code reviews.
  * 4x AI usage pool compared to Plus.
  * Standalone access to Grok Bot otherwise requires an independent X Premium+ subscription, making Cursor Premium a highly effective subscription consolidation target.
* **Personal Subscription Stack**:
  * Claude Pro ($20/mo): Primary code planning, architecture reviews, and Sonnet/Opus reasoning via Claude Code.
  * ChatGPT Plus ($20/mo): OpenAI Operator / Dots computer-use and general GPT-4o inference.
  * Gemini Pro (AGY / Google One): 2M context deep research and multimodal vision.
* **Work Separation Boundary**:
  * GitHub Copilot is restricted strictly to employer/work codebases; never routed to Project Sanctuary.

---

## 4. Regional Geo-Availability Boundaries

* **Claude Duet (Anthropic)**: Currently limited to US-based enterprise preview workspaces; non-US IP addresses and non-US corporate Google/Microsoft tenancies cannot activate Duet connectors.
* **Dots / Operator (OpenAI)**: Rolling preview restricted primarily to US ChatGPT Plus/Pro accounts. Users outside the US may encounter waitlist or geo-blocking constraints.
* **Grok Bot (xAI)**: Available globally through Cursor Premium ($30/mo) or X Premium+, with outbound phone calling dependent on regional telecom interconnects.
* **Gemini Agent / Project Astra (Google)**: Broadest international availability across Google One AI Premium and Google AI Studio.
