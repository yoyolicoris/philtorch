# Copilot instructions for `philtorch`

The shared instructions for every coding agent, Copilot included, are in
[`AGENTS.md`](../AGENTS.md) at the repository root: commands, architecture,
conventions, tests and branches. Add only Copilot-specific notes here.

## Terminal usage (avoid hangs)

- For quick Python checks, use one-liner `pixi run python -c "code"`; avoid `pixi run python - << 'PY'` heredocs and do not add `timeout=15000` wrappers, as they leave the terminal appearing finished without returning.

<!-- mermaid-ai-skills:start -->
## Mermaid Diagrams

When the user asks to create, edit, or visualize a diagram, follow the
instructions in `.github/instructions/mermaid.instructions.md`.
<!-- mermaid-ai-skills:end -->
