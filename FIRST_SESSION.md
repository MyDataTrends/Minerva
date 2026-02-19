# First Claude Code Session — Paste This

## Setup (one-time, in your terminal before starting Claude Code)

```bash
# 1. Push repo to GitHub (if not already done)
cd /path/to/Assay
git init
git add .
git commit -m "Initial public commit"
gh repo create assay --public --source=. --push
# (or use GitHub web UI to create repo, then git remote add + push)

# 2. Install Claude Code (requires Node.js 18+)
npm install -g @anthropic-ai/claude-code

# 3. Start Claude Code in your repo
cd /path/to/Assay
claude
```

## First Session Prompt

Paste this into Claude Code once it starts:

---

Read CLAUDE.md for full project context. Then do the following in order:

**Task 1: Build the Marketing Agent**

Create `agents/marketing.py` following the BaseAgent pattern in `agents/base.py`. This agent should:

- Inherit from BaseAgent with name="marketing", trigger_type=TriggerType.CRON
- Use OperationalMemory for state tracking
- Have a `run()` method that:
  1. Reads recent git log (last 7 days) and generates a changelog summary
  2. Reads any new Productizer outputs from the knowledge base
  3. Generates draft social media posts (HN, Reddit, Twitter) from changelog + productizer content
  4. Saves drafts to `agents/digests/marketing/` for human review
  5. Returns AgentResult with drafts as escalations (Priority.REVIEW)
- All LLM calls use `llm_manager.llm_interface.get_llm_completion()` with empty-string fallback handling
- Register it in `agents/cli.py` alongside the other agents
- Add it to `agents/agents_config.yaml`
- Write tests in `tests/test_marketing_agent.py`

**Task 2: Build the Support Agent**

Create `agents/support.py` following the same pattern. This agent should:

- Extend the Advocate's capabilities (read `agents/advocate.py` for reference)
- Maintain a FAQ knowledge base in `agents/knowledge_base/faq.json`
- When triggered by a GitHub issue or user question:
  1. Search the FAQ for matching answers (fuzzy string matching)
  2. Search the vector store (`learning/vector_store.py`) for relevant past interactions
  3. If high-confidence match found: draft a response (Priority.FYI — auto-handled)
  4. If low-confidence: escalate to human (Priority.REVIEW)
- Include a method to add new FAQ entries from resolved issues
- Register in cli.py and agents_config.yaml
- Write tests in `tests/test_support_agent.py`

**Task 3: Build the Telemetry Agent**

Create `agents/telemetry.py`. This agent should:

- Read from the InteractionLogger database (`llm_learning/interaction_logger.py`)
- Compute weekly usage metrics: total queries, most-used tools, success rates, popular intents
- Identify trends: growing/declining feature usage, error rate changes
- Feed insights back to Productizer via knowledge base artifacts
- Generate a weekly telemetry report in `agents/digests/telemetry/`
- Register in cli.py and agents_config.yaml
- Write tests in `tests/test_telemetry_agent.py`

After building each agent, run `pytest tests/test_<agent>_agent.py -v` to verify. Then run `pytest tests/ -v` to make sure nothing is broken.

---
