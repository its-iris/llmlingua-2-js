// SPDX-License-Identifier: MIT

import { AutoTokenizer } from "@huggingface/transformers";
import { LLMLingua2 } from "../../../src/index.js";
import { EXAMPLES } from "../long-texts.js";

// Xenova/gpt-4o is the Hugging Face equivalent of OpenAI's tiktoken with o200k_base.
// You can also use tiktoken or js-tiktoken directly here, but HF tokenizers are relatively fasts since 4.0
const oai_tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt-4o");
const modelName = "Arcoldd/llmlingua4j-bert-base-onnx";

const { promptCompressor } = await LLMLingua2.WithBERTMultilingual(modelName, {
  transformersJSConfig: {
    device: "cpu",
    dtype: "fp32",
  },
  oaiTokenizer: oai_tokenizer,
  modelOptions: {
    subfolder: "",
  },
});

const start = performance.now();

const result = await promptCompressor.compress_prompt(
  EXAMPLES[EXAMPLES.length - 1],
  {
    rate: 0.5,
  },
);

const end = performance.now();

console.log({ result });

console.log("Time taken for compression:", end - start, "ms");
console.log(
  "Time taken for compression (human-readable):",
  ((end - start) / 1000).toFixed(2),
  "seconds",
);
