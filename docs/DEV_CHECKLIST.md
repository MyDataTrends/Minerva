# Developer Checklist

Use this checklist to ensure consistency and quality when adding new features to Assay.

## 1. General Hygiene (Crucial)

- [ ] **Update Documentation**:
  - [ ] Update `README.md` if high-level features change.
  - [ ] Update relevant files in `docs/` or create new ones for complex features.
- [ ] **Artifact Management**:
  - [ ] Check if your code generates new files (reports, logs, data).
  - [ ] Add these patterns to `.gitignore` to prevent repo bloat.
- [ ] **Security**:
  - [ ] **NEVER** commit API keys or secrets.
  - [ ] Use `config/__init__.py` to read from `.env`.

## 2. Configuration

- [ ] **Environment Variables**:
  - [ ] Add new variables to `config/__init__.py`.
  - [ ] Provide sensible defaults (or raise error if required).
  - [ ] Use helper functions `get_env`, `get_bool`, `get_int`.
- [ ] **File Paths**:
  - [ ] Use `pathlib.Path` for filesystem operations.
  - [ ] Use `utils.security.secure_join` when handling user-provided paths.

## 3. Agents

- [ ] **Implementation**:
  - [ ] Inherit from `agents.base.BaseAgent`.
  - [ ] Implement `run(**kwargs) -> AgentResult`.
  - [ ] Handle exceptions and set `result.success = False`.
- [ ] **Registration**:
  - [ ] Add your agent class to `_AGENT_CLASSES` in `agents/cli.py`.
  - [ ] Add default configuration in `agents/config.py` (if applicable).
- [ ] **Logging**:
  - [ ] Get a logger using `utils.logging.get_logger(f"agents.{self.name}")`.

## 4. MCP Integration (Claude Desktop)

- [ ] **Tool Definition**:
  - [ ] Create tool class in `mcp_server/tools/`.
  - [ ] Inherit from `BaseTool`.
  - [ ] Define `name`, `description`, `category`.
  - [ ] Implement `get_parameters()` and `execute()`.
- [ ] **Registration**:
  - [ ] Register the tool with its category (e.g., `llm_category.register(MyTool())`).
  - [ ] Ensure the category is registered with `register_category`.

## 5. Testing

- [ ] **Unit Tests**:
  - [ ] Create a test file in `tests/` named `test_<feature>.py`.
  - [ ] Write tests to verify success and failure modes.
  - [ ] Run `pytest tests/test_<feature>.py` to verify.
