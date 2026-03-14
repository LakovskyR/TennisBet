import { OpenRouter } from "@openrouter/sdk";

const openrouter = new OpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY || "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"
});

// Stream the response to get reasoning tokens in usage
const stream = await openrouter.chat.send({
  chatGenerationParams: {
    model: "openrouter/hunter-alpha",
    messages: [
      {
        role: "user",
        content: "what are you?"
      }
    ],
    stream: true
  }
});

let response = "";
for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) {
    response += content;
    process.stdout.write(content);
  }

  // Usage information comes in the final chunk
  if (chunk.usage) {
    console.log("\nReasoning tokens:", chunk.usage.reasoningTokens);
  }
}
