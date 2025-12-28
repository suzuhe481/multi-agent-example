import type { ToolCall } from "@langchain/core/messages/tool";
import { arithmeticToolsByName, type ArithmeticToolName } from "./tools.js";

// Function to call a tool used ONLY by the arithmetic agent
export async function callTool(toolCall: ToolCall) {
  const name = toolCall.name as ArithmeticToolName;
  const tool = arithmeticToolsByName[name]; // Used by the code

  if (!tool) {
    throw new Error(`Unknown tool: ${toolCall.name}`);
  }

  return tool.invoke(toolCall);
}
