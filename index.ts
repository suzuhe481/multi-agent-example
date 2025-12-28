import { runManager } from "./agents/managerAgent/index.js";
import type { BaseMessage } from "@langchain/core/messages";

/**  Formats and prints the trace, differentiating between human,
 * AI, and tool messages.
 * eg.
 * [human]: What is  30 - seven?
 * [ai]:
 * [tool]: 23
 */
function printTrace(messages: BaseMessage[]) {
  for (const msg of messages) {
    if (msg.type === "human") {
      console.log(`[human]: ${msg.text}`);
    } else if (msg.type === "ai") {
      console.log(`[ai]: ${msg.text}`);
    } else if (msg.type === "tool") {
      console.log(`[tool]: ${msg.text}`);
    }
  }
}

const userInput = process.argv.slice(2).join(" ");

if (!userInput) {
  console.error("Please provide a query.");
  process.exit(1);
}

// Starting point
// Runs the manager agent with the user's input.
(async () => {
  try {
    const result = await runManager(userInput);

    if (result.type === "error") {
      console.log(`[system]: ${result.message}`);
    } else {
      printTrace(result.messages);
    }
  } catch (error) {
    console.log(error);
    process.exit(1);
  }
})();
