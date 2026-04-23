import { LLMLingua2 } from "@its-iris/llmlingua-2";
import { AutoTokenizer } from "@huggingface/transformers";

export const LLMLingua2CompressorModels = {
  TINYBERT: {
    modelName: "atjsh/llmlingua-2-js-tinybert-meetingbank",
    transformersJSConfig: {
      device: "auto",
      dtype: "fp32",
    },
    llmlingua2Config: {
      maxBatchSize: 50,
      maxForceTokens: 100,
      maxSequenceLength: 312,
    },
    factory: LLMLingua2.WithBERTMultilingual,
  },
  MOBILEBERT: {
    modelName: "atjsh/llmlingua-2-js-mobilebert-meetingbank",
    transformersJSConfig: {
      device: "auto",
      dtype: "fp32",
    },
    llmlingua2Config: {
      maxBatchSize: 50,
      maxForceTokens: 100,
      maxSequenceLength: 128,
    },
    factory: LLMLingua2.WithBERTMultilingual,
  },
  BERT: {
    modelName: "Arcoldd/llmlingua4j-bert-base-onnx",
    transformersJSConfig: {
      device: "auto",
      dtype: "fp32",
    },
    modelOptions: { subfolder: "" },
    factory: LLMLingua2.WithBERTMultilingual,
  },
  ROBERTA: {
    modelName: "atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank",
    transformersJSConfig: {
      device: "gpu",
      dtype: "int8",
    },
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

  const { promptCompressor } = await modelConfig.factory(
    modelConfig.modelName,
    {
      transformersJSConfig: modelConfig.transformersJSConfig,
      oaiTokenizer: oai_tokenizer,
      modelOptions: modelConfig.modelOptions,
      llmlingua2Config: modelConfig.llmlingua2Config,
    },
  );
  return promptCompressor;
}
