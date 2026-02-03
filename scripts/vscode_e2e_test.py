#!/usr/bin/env python3
"""VS Code MCP Server E2E Testing Guide.

This script provides instructions and utilities for end-to-end testing
of the gemini-research-mcp server with VS Code and Claude Desktop.

Run with: uv run python scripts/vscode_e2e_test.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_mcp_config_for_vscode() -> dict:
    """Generate MCP server configuration for VS Code settings.json."""
    project_root = get_project_root()
    
    return {
        "mcpServers": {
            "gemini-research": {
                "command": "uv",
                "args": [
                    "run",
                    "--directory",
                    str(project_root),
                    "gemini-research-mcp"
                ],
                "env": {
                    "GEMINI_API_KEY": "${env:GEMINI_API_KEY}"
                }
            }
        }
    }


def get_claude_desktop_config() -> dict:
    """Generate MCP server configuration for Claude Desktop."""
    project_root = get_project_root()
    
    return {
        "mcpServers": {
            "gemini-research": {
                "command": "uv",
                "args": [
                    "run",
                    "--directory",
                    str(project_root),
                    "gemini-research-mcp"
                ],
                "env": {
                    "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
                }
            }
        }
    }


def test_server_startup():
    """Test that the server starts without errors."""
    print("🧪 Testing server startup...")
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", """
import asyncio
from gemini_research_mcp.server import mcp, lifespan

async def test():
    async with lifespan(mcp):
        tools = await mcp.list_tools()
        print(f"✅ Server started successfully with {len(tools)} tools")
        for t in tools:
            print(f"   - {t.name}")

asyncio.run(test())
"""],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=get_project_root(),
        )
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ Server startup failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Server startup timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_mcp_inspector():
    """Test using MCP Inspector if available."""
    print("\n🔍 MCP Inspector Test")
    print("=" * 50)
    print("""
To test with MCP Inspector:

1. Install MCP Inspector:
   npx @anthropic/mcp-inspector

2. Run with your server:
   npx @anthropic/mcp-inspector uv run --directory . gemini-research-mcp

3. In the inspector:
   - Verify all 9 tools are listed
   - Test 'list_format_templates' (no API key needed)
   - Test 'fetch_webpage' with a public URL
   - Test 'research_web' with API key
""")


def print_vscode_instructions():
    """Print VS Code configuration instructions."""
    print("\n📝 VS Code MCP Configuration")
    print("=" * 50)
    
    config = get_mcp_config_for_vscode()
    
    print("""
To configure VS Code to use this MCP server:

1. Open VS Code Settings (Cmd+, on macOS)
2. Search for "mcp"
3. Click "Edit in settings.json"
4. Add the following configuration:
""")
    
    print(json.dumps(config, indent=2))
    
    print("""
5. Set your GEMINI_API_KEY environment variable:
   export GEMINI_API_KEY="your-api-key"

6. Restart VS Code or reload the window

7. Test by asking Copilot:
   "Use gemini-research to search for Python best practices"
""")


def print_claude_desktop_instructions():
    """Print Claude Desktop configuration instructions."""
    print("\n🖥️  Claude Desktop Configuration")
    print("=" * 50)
    
    config = get_claude_desktop_config()
    
    print("""
To configure Claude Desktop to use this MCP server:

1. Open Claude Desktop config file:
   - macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
   - Windows: %APPDATA%\\Claude\\claude_desktop_config.json

2. Add the following configuration:
""")
    
    print(json.dumps(config, indent=2))
    
    print("""
3. Restart Claude Desktop

4. Test by asking Claude:
   "What tools do you have available?"
   
   Then test with:
   "Use research_web to find the latest news about Python 3.13"
""")


def print_e2e_test_scenarios():
    """Print recommended e2e test scenarios."""
    print("\n🎯 E2E Test Scenarios")
    print("=" * 50)
    print("""
Test these scenarios to validate the MCP server works correctly:

## Scenario 1: Tool Discovery (No API Key)
Ask: "What tools does gemini-research provide?"
Expected: Should list all 9 tools with descriptions

## Scenario 2: List Format Templates (No API Key)
Ask: "Show me the available research format templates"
Expected: Should return JSON list of templates

## Scenario 3: Fetch Webpage (No API Key)
Ask: "Fetch the content from https://httpbin.org/html"
Expected: Should return extracted HTML content

## Scenario 4: Quick Research (Requires API Key)
Ask: "Use research_web to explain what OMOP CDM is"
Expected: Should return grounded search results with sources

## Scenario 5: Deep Research (Requires API Key)
Ask: "Use research_deep to compare Python web frameworks"
Expected: Should start autonomous research, show progress

## Scenario 6: Elicitation (Requires API Key + Supporting Client)
Ask: "Research best practices" (vague query)
Expected: Should prompt for clarifying questions

## Scenario 7: Export (After Deep Research)
Ask: "Export the last research session to markdown"
Expected: Should return downloadable markdown file
""")


def run_automated_validation():
    """Run automated validation checks."""
    print("\n🔄 Automated Validation")
    print("=" * 50)
    
    checks = []
    
    # Check 1: Imports
    print("\n1. Checking imports...")
    try:
        from gemini_research_mcp.server import mcp
        from fastmcp import Context, FastMCP
        checks.append(("Imports", True))
        print("   ✅ All imports successful")
    except ImportError as e:
        checks.append(("Imports", False))
        print(f"   ❌ Import error: {e}")
    
    # Check 2: Server type
    print("\n2. Checking server type...")
    try:
        from fastmcp import FastMCP
        from gemini_research_mcp.server import mcp
        if isinstance(mcp, FastMCP):
            checks.append(("Server Type", True))
            print("   ✅ Server is FastMCP instance")
        else:
            checks.append(("Server Type", False))
            print(f"   ❌ Server type: {type(mcp)}")
    except Exception as e:
        checks.append(("Server Type", False))
        print(f"   ❌ Error: {e}")
    
    # Check 3: Tool count
    print("\n3. Checking tool registration...")
    try:
        import asyncio
        from gemini_research_mcp.server import mcp
        
        async def count_tools():
            return await mcp.list_tools()
        
        tools = asyncio.run(count_tools())
        if len(tools) == 9:
            checks.append(("Tool Count", True))
            print(f"   ✅ {len(tools)} tools registered")
        else:
            checks.append(("Tool Count", False))
            print(f"   ❌ Expected 9 tools, got {len(tools)}")
    except Exception as e:
        checks.append(("Tool Count", False))
        print(f"   ❌ Error: {e}")
    
    # Check 4: Elicit signature
    print("\n4. Checking elicit() API...")
    try:
        import inspect
        from fastmcp import Context
        
        sig = inspect.signature(Context.elicit)
        params = list(sig.parameters.keys())
        
        if "response_type" in params and "schema" not in params:
            checks.append(("Elicit API", True))
            print("   ✅ elicit() uses response_type parameter")
        else:
            checks.append(("Elicit API", False))
            print(f"   ❌ elicit() params: {params}")
    except Exception as e:
        checks.append(("Elicit API", False))
        print(f"   ❌ Error: {e}")
    
    # Check 5: TaskSupport
    print("\n5. Checking TaskSupport...")
    try:
        import asyncio
        from gemini_research_mcp.server import mcp, lifespan
        
        async def check_task_support():
            async with lifespan(mcp):
                from gemini_research_mcp.server import _task_support
                return _task_support is not None
        
        if asyncio.run(check_task_support()):
            checks.append(("TaskSupport", True))
            print("   ✅ TaskSupport initializes correctly")
        else:
            checks.append(("TaskSupport", False))
            print("   ❌ TaskSupport is None")
    except Exception as e:
        checks.append(("TaskSupport", False))
        print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Summary")
    print("=" * 50)
    
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    
    for name, ok in checks:
        status = "✅" if ok else "❌"
        print(f"   {status} {name}")
    
    print(f"\n   {passed}/{total} checks passed")
    
    return passed == total


def main():
    """Main entry point."""
    print("🚀 Gemini Research MCP Server - E2E Testing Guide")
    print("=" * 60)
    
    # Run automated validation
    all_passed = run_automated_validation()
    
    # Print instructions
    print_vscode_instructions()
    print_claude_desktop_instructions()
    test_mcp_inspector()
    print_e2e_test_scenarios()
    
    # Final status
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All automated checks passed!")
        print("   Proceed with manual VS Code/Claude Desktop testing.")
    else:
        print("❌ Some checks failed. Fix issues before e2e testing.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
