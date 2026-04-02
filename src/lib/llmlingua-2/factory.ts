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
  PretrainedConfig
} from "@huggingface/transformers";

// This is no longer exposed directly after Transformers 4.x
type TransformersJSConfig = NonNullable<PretrainedConfig['transformers.js_config']>;

import { PromptCompressorLLMLingua2 } from "./prompt-compressor.js";
import {
  get_pure_tokens_bert_base_multilingual_cased,
  get_pure_tokens_xlm_roberta_large,
  is_begin_of_new_word_bert_base_multilingual_cased,
  is_begin_of_new_word_xlm_roberta_large,
  Logger,
} from "./utils.js";

type PreTrainedTokenizerOptions = Parameters<
  typeof AutoTokenizer.from_pretrained
>[1];
type PretrainedModelOptions = Parameters<
  typeof AutoModelForTokenClassification.from_pretrained
>[1];

async function prepareDependencies(
  modelName: string,
  transformersJSConfig: TransformersJSConfig,
  logger: Logger,

  pretrainedConfig?: PretrainedConfig | null,
  pretrainedTokenizerOptions?: PreTrainedTokenizerOptions | null,
  modelSpecificOptions?: PretrainedModelOptions | null
) {
  const config =
    pretrainedConfig ?? (await AutoConfig.from_pretrained(modelName));
  logger({ config });

  const tokenizerConfig = {
    config: {
      ...config,
      ...(transformersJSConfig
        ? { "transformers.js_config": transformersJSConfig }
        : {}),
    },
    ...pretrainedTokenizerOptions,
  };
  logger({ tokenizerConfig });

  const tokenizer = await AutoTokenizer.from_pretrained(
    modelName,
    tokenizerConfig
  );
  logger({ tokenizer });

  const modelConfig = {
    config: {
      ...config,
      ...(transformersJSConfig
        ? { "transformers.js_config": transformersJSConfig }
        : {}),
    },
    ...modelSpecificOptions,
  };
  logger({ modelConfig });

  const model = await AutoModelForTokenClassification.from_pretrained(
    modelName,
    modelConfig
  );
  logger({ model });

  return { model, tokenizer, config };
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
  oaiTokenizer: { encode: (text: string) => { length: number } },

  /**
   * Optional pretrained configuration.
   */
  pretrainedConfig?: PretrainedConfig | null;

  /**
   * Optional pretrained tokenizer options.
   */
  pretrainedTokenizerOptions?: PreTrainedTokenizerOptions | null;

  /**
   * Optional model-specific options.
   */
  modelSpecificOptions?: PretrainedModelOptions | null;

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

  /**
   * The configuration used for the model.
   */
  config: AutoConfig;
}

/**
 * Factory functions to create instances of LLMLingua-2 PromptCompressor
 * with XLM-RoBERTa model.
 *
 * @category Factory
 * 
 * @example 
 * ```ts
import { LLMLingua2 } from "@atjsh/llmlingua-2";

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
    modelSpecificOptions: {
      use_external_data_format: true,
    },
  }
);

const compressedText: string = await promptCompressor.compress_prompt(
  "LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance.",
  { rate: 0.8 }
);

console.log({ compressedText });
```
 */
export async function WithXLMRoBERTa(
  modelName: string,
  options: LLMLingua2FactoryOptions
): Promise<LLMLingua2FactoryReturn> {
  const {
    transformersJSConfig,
    oaiTokenizer,
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions,
    logger = console.log,
  } = options;

  const { model, tokenizer, config } = await prepareDependencies(
    modelName,
    transformersJSConfig,
    logger,
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions
  );

  const promptCompressor = new PromptCompressorLLMLingua2(
    model,
    tokenizer,
    get_pure_tokens_xlm_roberta_large,
    is_begin_of_new_word_xlm_roberta_large,
    oaiTokenizer
  );

  logger({ promptCompressor });

  return {
    promptCompressor,
    model,
    tokenizer,
    config,
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
import { LLMLingua2 } from "@atjsh/llmlingua-2";

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
    modelSpecificOptions: {
      subfolder: "",
    },
  }
);

const compressedText: string = await promptCompressor.compress_prompt(
  "LLMLingua-2, a small-size yet powerful prompt compression method trained via data distillation from GPT-4 for token classification with a BERT-level encoder, excels in task-agnostic compression. It surpasses LLMLingua in handling out-of-domain data, offering 3x-6x faster performance.",
  { rate: 0.8 }
);

console.log({ compressedText });
```
 */
export async function WithBERTMultilingual(
  modelName: string,
  options: LLMLingua2FactoryOptions
): Promise<LLMLingua2FactoryReturn> {
  const {
    transformersJSConfig,
    oaiTokenizer,
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions,
    logger = console.log,
  } = options;

  const { model, tokenizer, config } = await prepareDependencies(
    modelName,
    transformersJSConfig,
    logger,
    pretrainedConfig,
    pretrainedTokenizerOptions,
    modelSpecificOptions
  );

  const promptCompressor = new PromptCompressorLLMLingua2(
    model,
    tokenizer,
    get_pure_tokens_bert_base_multilingual_cased,
    is_begin_of_new_word_bert_base_multilingual_cased,
    oaiTokenizer
  );

  logger({ promptCompressor });

  return {
    promptCompressor,
    model,
    tokenizer,
    config,
  };
}
