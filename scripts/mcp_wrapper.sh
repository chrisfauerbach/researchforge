#!/bin/bash
# Wrapper for Claude Desktop MCP integration.
# Set the path in Claude Desktop's MCP config to this script.
# The -T flag disables pseudo-TTY (required for stdio transport).
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec docker compose -f "$SCRIPT_DIR/docker-compose.yml" exec -T app python -m researchforge mcp
