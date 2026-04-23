// SPDX-License-Identifier: MIT

/**
 * @categoryDescription Factory
A collection of utility functions and types for model-specific token handling.
 *
 * @showCategories
 */

import {
  AutoConfig,
  AutoModelForTokenClassification,
  AutoTokenizer,
  PretrainedConfig,
} from "@huggingface/transformers";
import {
  LLMLingua2Config,
  PromptCompressorLLMLingua2,
} from "./prompt-compressor.js";
import {
  get_pure_tokens_bert_base_multilingual_cased,
  get_pure_tokens_xlm_roberta_large,
  is_begin_of_new_word_bert_base_multilingual_cased,
  is_begin_of_new_word_xlm_roberta_large,
  Logger,
  DEFAULT_LOGGER,
} from "./utils.js";

// The types below are not directly exposed by Transformers API
type TransformersJSConfig = NonNullable<
  PretrainedConfig["transformers.js_config"]
>;
type PreTrainedTokenizerOptions = Parameters<
  typeof AutoTokenizer.from_pretrained
>[1];
type PretrainedModelOptions = Parameters<
  typeof AutoModelForTokenClassification.from_pretrained
>[1];

/**
 * Builds the options objects for the model and the tokenizer, and loads them both.
 *
 * @category Factory
 */
async function prepareDependencies(
  modelName: string,
  options: LLMLingua2FactoryOptions,
) {
  const {
    transformersJSConfig,
    tokenizerOptions,
    modelOptions,
    logger = DEFAULT_LOGGER,
  } = options;
  const defaultConfig = await AutoConfig.from_pretrained(modelName);

  // Override defaultConfig if user provided a config for tokenizer or model
  // Always override transformers.js_config with transformersJSConfig
  const buildOpts = (
    opts?: PretrainedModelOptions | PreTrainedTokenizerOptions,
  ) => ({
    ...opts,
    config: {
      ...(opts?.config || defaultConfig),
      ...(transformersJSConfig && {
        "transformers.js_config": transformersJSConfig,
      }),
    },
  });

  // Build the option objects
  const finalTokenizerOptions = buildOpts(tokenizerOptions);
  const finalModelOptions = buildOpts(modelOptions);
  logger({
    defaultConfig: defaultConfig,
    tokenizerConfig: finalTokenizerOptions.config,
    modelConfig: finalModelOptions.config,
  });

  // Load the models asynchronously
  const [tokenizer, model] = await Promise.all([
    AutoTokenizer.from_pretrained(modelName, finalTokenizerOptions),
    AutoModelForTokenClassification.from_pretrained(
      modelName,
      finalModelOptions,
    ),
  ]);
  logger({ tokenizer, model });

  return { model, tokenizer };
}

/**
 * Options for the LLMLingua-2 factory functions.
 *
 * @category Factory
 */
export interface LLMLingua2FactoryOptions {
  /**
   * Configuration for Transformers.js.
   */
  transformersJSConfig: TransformersJSConfig;

  /**
   * The tokenizer to use calculating the compression rate.
   * It needs to return a collection that exposes length.
   */
  oaiTokenizer: { encode: (text: string) => { length: number } };

  /**
   * Optional LLMLingua-2 configuration.
   */
  llmlingua2Config?: LLMLingua2Config;

  /**
   * Optional pretrained tokenizer options.
   * This does not refer to the oaiTokenizer!
   */
  tokenizerOptions?: PreTrainedTokenizerOptions;

  /**
   * Optional model-specific options.
   */
  modelOptions?: PretrainedModelOptions;

  /**
   * Optional logger function.
   */
  logger?: Logger;
}

/**
 * Return type for the LLMLingua-2 factory functions. Use `promptCompressor` to compress prompts.
 *
 * @category Factory
 */
export interface LLMLingua2FactoryReturn {
  /**
   * Instance of LLMLingua-2 PromptCompressor.
   *
   * @see {@link PromptCompressorLLMLingua2}
   */
  promptCompressor: PromptCompressorLLMLingua2;

  /**
   * The model used for token classification.
   */
  model: AutoModelForTokenClassification;

  /**
   * The tokenizer used for tokenization.
   */
  tokenizer: AutoTokenizer;
}

/**
 * Factory functions to create instances of LLMLingua-2 PromptCompressor
 * with XLM-RoBERTa model.
 *
 * @category Factory
 * 
 * @example 
 * ```ts
import { LLMLingua2 } from "@its-iris/llmlingua-2";

import { AutoTokenizer } from "@huggingface/transformers";

const modelName = "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank";
const oai_tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt-4o");

const { promptCompressor } = await LLMLingua2.WithXLMRoBERTa(modelName,
  {
    transformersJSConfig: {
      device: "auto",
      dtype: "fp32",
    },
    oaiTokenizer: oai_tokenizer,
    modelOptions: {
      use_external_data_format: true,
    },
  }
);

const compressedText: string = await promptCompressor.compress(
  "LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance.",
  { rate: 0.8 }
);

console.log({ compressedText });
```
 */
export async function WithXLMRoBERTa(
  modelName: string,
  options: LLMLingua2FactoryOptions,
): Promise<LLMLingua2FactoryReturn> {
  const { oaiTokenizer, llmlingua2Config, logger = DEFAULT_LOGGER } = options;

  const { model, tokenizer } = await prepareDependencies(modelName, options);

  const promptCompressor = new PromptCompressorLLMLingua2(
    model,
    tokenizer,
    get_pure_tokens_xlm_roberta_large,
    is_begin_of_new_word_xlm_roberta_large,
    oaiTokenizer,
    llmlingua2Config,
    logger,
  );
  logger({ promptCompressor });

  return {
    promptCompressor,
    model,
    tokenizer,
  };
}

/**
 * Factory functions to create instances of LLMLingua-2 PromptCompressor
 * with BERT Multilingual model.
 *
 * @category Factory
 * 
 * @example 
 * ```ts
import { LLMLingua2 } from "@its-iris/llmlingua-2";

import { AutoTokenizer } from "@huggingface/transformers";

const modelName = "Arcoldd/llmlingua4j-bert-base-onnx";
const oai_tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt-4o");

const { promptCompressor } = await LLMLingua2.WithBERTMultilingual(modelName,
  {
    transformersJSConfig: {
      device: "auto",
      dtype: "fp32",
    },
    oaiTokenizer: oai_tokenizer,
    modelOptions: {
      subfolder: "",
    },
  }
);

const compressedText: string = await promptCompressor.compress(
  "LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance.",
  { rate: 0.8 }
);

console.log({ compressedText });
```
 */
export async function WithBERTMultilingual(
  modelName: string,
  options: LLMLingua2FactoryOptions,
): Promise<LLMLingua2FactoryReturn> {
  const { oaiTokenizer, llmlingua2Config, logger = DEFAULT_LOGGER } = options;

  const { model, tokenizer } = await prepareDependencies(modelName, options);

  const promptCompressor = new PromptCompressorLLMLingua2(
    model,
    tokenizer,
    get_pure_tokens_bert_base_multilingual_cased,
    is_begin_of_new_word_bert_base_multilingual_cased,
    oaiTokenizer,
    llmlingua2Config,
    logger,
  );
  logger({ promptCompressor });

  return {
    promptCompressor,
    model,
    tokenizer,
  };
}
