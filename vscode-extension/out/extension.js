"use strict";
/**
 * Gemini Research MCP - VS Code Extension
 *
 * This is a thin wrapper extension that registers the MCP server
 * for use with GitHub Copilot Chat. The actual functionality is
 * provided by the gemini-research-mcp Python package via uvx.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
function activate(context) {
    console.log("Gemini Research MCP extension activated");
    // The MCP server is automatically registered via package.json contributes.mcpServers
    // No additional activation code is needed - VS Code handles the MCP lifecycle.
}
function deactivate() {
    console.log("Gemini Research MCP extension deactivated");
}
//# sourceMappingURL=extension.js.map