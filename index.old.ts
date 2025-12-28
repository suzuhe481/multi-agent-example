import "dotenv/config";
// import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatOllama } from "@langchain/ollama";
import { tool } from "@langchain/core/tools";
import * as z from "zod";

/* ===== STEP 1: Define tools and model ===== */

// Create the model
// const model = new ChatGoogleGenerativeAI({
//   // model: "gemini-2.5-flash",
//   model: "gemini-2.0-flash",
//   // maxOutputTokens: 2048,
//   temperature: 0,
// });
const model = new ChatOllama({
  model: "qwen2.5:3b",
  // model: "qwen3:4b", // Slow and sometimes doesn't finish
  temperature: 0, // Controls randomness. 0 is good for deterministic results (Math, tools, agents).
});

// Define tools
const add = tool(({ a, b }) => a + b, {
  name: "add",
  description: `
    Adds two numbers together.
    Use this ONLY when the user asks for addition
    between exactly two numeric values.
    Do NOT use for subtraction, multiplication, or word problems.
    `,
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const subtract = tool(({ a, b }) => a - b, {
  name: "subtract",
  description: "Subtract two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const multiply = tool(({ a, b }) => a * b, {
  name: "multiply",
  description: "Multiply two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

const divide = tool(({ a, b }) => a / b, {
  name: "divide",
  description: "Divide two numbers",
  schema: z.object({
    a: z.number().describe("First number"),
    b: z.number().describe("Second number"),
  }),
});

// Augment the LLM with tools
const toolsByName = {
  [add.name]: add,
  [subtract.name]: subtract,
  [multiply.name]: multiply,
  [divide.name]: divide,
};

const tools = Object.values(toolsByName);
const modelWithTools = model.bindTools(tools); // Lets LLM see what tools exist

/* ===== STEP 2: Define model node ===== */
// THe model node is used to call the LLM and decide whether to call a tool or not.
import { task } from "@langchain/langgraph";
import {
  AIMessage,
  BaseMessage,
  SystemMessage,
} from "@langchain/core/messages";

const callLlm = task({ name: "callLlm" }, async (messages: BaseMessage[]) => {
  // Guardrail 1 - Prompt level: Adding explicit instructions
  return modelWithTools.invoke([
    // System message include instructions.
    // Guardrails at the prompt level with explicit instructions, constraints, and tool access.
    new SystemMessage(
      `
      You are a helpful teacher tasked with answering simple questions up to a high school grade level ranging from multiple subjects such as
      arithmetic, geography, science, history, and more. You should respond with short, simple, and clear answers.

      Rules:
      - If the user asks for simple arithmetic questions involving an operation between two numbers, use tools.
      - If the question does not require tools, answer directly.
      - If the request is unclear, ask for clarification.
       
      Tool constraints:
      - You only have access to the following tools
      ${tools.map((tool) => `- ${tool.name}`).join("\n")}
      `
    ),
    ...messages,
  ]);
});

/* ===== STEP 3: Define tool node ===== */
// The tool node is used to call the tools and return the results.
import type { ToolCall } from "@langchain/core/messages/tool";

const callTool = task({ name: "callTool" }, async (toolCall: ToolCall) => {
  const tool = toolsByName[toolCall.name]; // Used by the code

  if (!tool) {
    throw new Error(`Unknown tool: ${toolCall.name}`);
  }

  return tool.invoke(toolCall);
});

/* ===== STEP 4: Define agent ===== */
import { addMessages, entrypoint } from "@langchain/langgraph";

// import type { isAIMessage } from "@langchain/core/messages";

const agent = entrypoint({ name: "agent" }, async (messages: BaseMessage[]) => {
  let modelResponse = await callLlm(messages);

  while (true) {
    // Model answered directly (no tools) - This shouldn't be used in final version since the LLM overrides
    // tool calls, which can be incorrect.
    if (!modelResponse.tool_calls?.length) {
      messages = addMessages(messages, [modelResponse]);
      break;
    }

    // Guardrail - Model did nothing useful (No tool calls and no content)
    // if (
    //   !modelResponse.tool_calls?.length &&
    //   (!modelResponse.content || modelResponse.content.length === 0)
    // ) {
    //   messages = addMessages(messages, [
    //     modelResponse,
    //     new AIMessage("I'm not sure how to handle that. Could you rephase?"),
    //   ]);
    // }

    // Normal exit
    // Tool only enforcement
    // Stops the loop if there are no tool calls
    // Very important. If you don't do this, the loop will run forever, wasting tokens.
    // But not needed when LLM answers directly.
    // if (!modelResponse.tool_calls?.length) {
    //   break;
    // }

    // Execute tools
    const toolResults = await Promise.all(
      modelResponse.tool_calls.map((toolCall) => callTool(toolCall))
    );
    messages = addMessages(messages, [modelResponse, ...toolResults]);
    modelResponse = await callLlm(messages);
  }

  return messages;
});

// Invoke
import { HumanMessage } from "@langchain/core/messages";

// const result = await agent.invoke([new HumanMessage("Add 1 and 3?")]);

// for (const message of result) {
//   console.log(`[${message.type}]: ${message.text}`);
// }

const userInput = process.argv.slice(2).join(" ");

if (!userInput) {
  console.error("Please provide a query.");
  process.exit(1);
}

const result = await agent.invoke([new HumanMessage(userInput)]);
for (const message of result) {
  console.log(`[${message.type}]: ${message.text}`);
}
