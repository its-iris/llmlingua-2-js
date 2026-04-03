import {
  AutoTokenizer,
  AutoConfig,
  BertForTokenClassification,
  MobileBertPreTrainedModel,
} from "@huggingface/transformers";
import { LLMLingua2 } from "@its-iris/llmlingua-2";

class MobileBertForTokenClassification extends MobileBertPreTrainedModel {
  async _call(model_inputs) {
    return await super._call(model_inputs);
  }
}

export const LLMLingua2CompressorModels = {
  TINYBERT: {
    modelName: "atjsh/llmlingua-2-js-tinybert-meetingbank",
    defaultDevice: "auto",
    defaultModelDataType: "fp32",
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 312,
    modelClass: BertForTokenClassification,
    tokenUtils: {
      getPureTokens: LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
      isBeginOfNewWord:
        LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    },
  },
  MOBILEBERT: {
    modelName: "atjsh/llmlingua-2-js-mobilebert-meetingbank",
    defaultDevice: "auto",
    defaultModelDataType: "fp32",
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 128,
    modelClass: MobileBertForTokenClassification,
    tokenUtils: {
      getPureTokens: LLMLingua2.get_pure_tokens_bert_base_multilingual_cased,
      isBeginOfNewWord:
        LLMLingua2.is_begin_of_new_word_bert_base_multilingual_cased,
    },
  },
  BERT: {
    modelName: "Arcoldd/llmlingua4j-bert-base-onnx",
    defaultDevice: "auto",
    defaultModelDataType: "fp32",
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 512,
    modelOptions: { subfolder: "" },
    factory: LLMLingua2.WithBERTMultilingual,
  },
  ROBERTA: {
    modelName: "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank",
    defaultDevice: "gpu",
    defaultModelDataType: "int8",
    maxBatchSize: 50,
    maxForceTokens: 100,
    maxSequenceLength: 512,
    modelOptions: { use_external_data_format: true },
    factory: LLMLingua2.WithXLMRoBERTa,
  },
};

export async function createCompressor(modelKey) {
  const oai_tokenizer = await AutoTokenizer.from_pretrained("Xenova/gpt-4o");
  const modelConfig = LLMLingua2CompressorModels[modelKey];

  if (!modelConfig) {
    throw new Error(`Model ${modelKey} not found.`);
  }

  const transformersJSConfig = {
    device: modelConfig.defaultDevice,
    dtype: modelConfig.defaultModelDataType,
  };

  if (modelConfig.factory) {
    const { promptCompressor } = await modelConfig.factory(
      modelConfig.modelName,
      {
        transformersJSConfig,
        oaiTokenizer: oai_tokenizer,
        modelOptions: modelConfig.modelOptions,
      },
    );
    return promptCompressor;
  } else if (modelConfig.modelClass) {
    const config = await AutoConfig.from_pretrained(modelConfig.modelName);
    const tokenizer = await AutoTokenizer.from_pretrained(
      modelConfig.modelName,
      {
        config: {
          ...config,
          "transformers.js_config": transformersJSConfig,
        },
      },
    );

    const model = await modelConfig.modelClass.from_pretrained(
      modelConfig.modelName,
      {
        config: {
          ...config,
          "transformers.js_config": transformersJSConfig,
        },
      },
    );

    return new LLMLingua2.PromptCompressor(
      model,
      tokenizer,
      modelConfig.tokenUtils.getPureTokens,
      modelConfig.tokenUtils.isBeginOfNewWord,
      oai_tokenizer,
      {
        max_batch_size: modelConfig.maxBatchSize,
        max_force_token: modelConfig.maxForceTokens,
        max_seq_length: modelConfig.maxSequenceLength,
      },
    );
  }

  throw new Error("Invalid model configuration.");
}
