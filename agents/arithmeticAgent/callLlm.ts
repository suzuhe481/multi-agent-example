import { SystemMessage } from "@langchain/core/messages";
import { arithmeticModel } from "./model.js";
import type { BaseMessage } from "@langchain/core/messages";

/**
 * Calls the arithmetic model with the given messages.
 * The arithmetic model is given instructions to follow
 * specific rules for answering arithmetic questions.
 * @param {BaseMessage[]} messages - The messages to be passed to the model.
 * @returns {Promise<BaseMessage[]>} - The response from the model.
 */
export async function callLlm(messages: BaseMessage[]) {
  return arithmeticModel.invoke([
    new SystemMessage(`
    You are a helpful teacher.

    Rules:
    - You MUST use tools for arithmetic.
    - You MUST NOT answer arithmetic directly.
    - If the question is not arithmetic, respond with:
      "I can only answer arithmetic questions."
    `),
    ...messages,
  ]);
}
