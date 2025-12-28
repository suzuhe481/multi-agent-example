import type { BaseMessage } from "@langchain/core/messages";

export function formatFinalOutput(messages: BaseMessage[]): string {
  // Find last AI message
  const lastAiMessage = [...messages].reverse().find((m) => m.type === "ai");

  if (!lastAiMessage) {
    return "No response generated.";
  }

  return lastAiMessage.text;
}
