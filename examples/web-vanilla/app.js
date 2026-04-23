import { createCompressor } from "./factory.js";

class CompressorApp extends HTMLElement {
  constructor() {
    super();
    this.innerHTML = `
            <div class="grid lg:grid-cols-2 gap-6 bg-white shadow rounded-lg p-6">
                <!-- Settings Panel -->
                <div class="space-y-4 border-gray-200 lg:border-r lg:pr-6">
                    
                    <div class="space-y-2">
                        <label class="block text-sm font-medium">Model</label>
                        <select id="model-select" class="w-full bg-gray-50 border p-2 rounded">                        
                            <option value="TINYBERT">atjsh/llmlingua-2-js-tinybert-meetingbank</option>    
                            <option value="MOBILEBERT">atjsh/llmlingua-2-js-mobilebert-meetingbank</option>
                            <option value="BERT">Arcoldd/llmlingua4j-bert-base-onnx</option>
                            <option value="ROBERTA">atjsh/llmlingua-2-js-xlm-roberta-large-meetingbank</option>
                        </select>
                    </div>

                    <div class="space-y-2">
                      <label class="block text-sm font-medium" for="rate">Target Compression Rate: <span id="rate-value">0.50</span></label>
                      <input id="rate" type="range" min="0.1" max="1.0" step="0.01" value="0.5" class="w-full">
                    </div>

                    <button id="compress-btn" class="w-full mt-4 bg-indigo-600 hover:bg-indigo-700 text-white p-3 rounded font-medium disabled:opacity-50">
                        Compress Text
                    </button>

                    <div id="status" class="text-sm mt-2 text-gray-500 font-mono text-center">Ready</div>

                    <div id="download-progress" class="mt-3 opacity-0 max-h-0 overflow-hidden transition-all duration-300">
                      <div class="flex justify-between text-xs text-gray-600 mb-1">
                        <span>Downloading model</span>
                        <span id="download-progress-value"></span>
                      </div>
                      <div class="w-full h-2 bg-gray-200 rounded overflow-hidden">
                        <div id="download-progress-bar" class="h-full bg-indigo-500 transition-all duration-150" style="width: 0%"></div>
                      </div>
                    </div>
                </div>

                <!-- I/O Panel -->
                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium mb-1 flex justify-between">
                            Original Text <span id="original-count" class="text-gray-400">0 chars</span>
                        </label>
                        <textarea id="original-text" rows="8" class="w-full border rounded p-3 font-mono text-sm" placeholder="Paste your text here..."></textarea>
                    </div>

                    <div>
                        <label class="block text-sm font-medium mb-1 flex justify-between">
                            Compressed Text <span id="compressed-count" class="text-gray-400">0 chars</span>
                        </label>
                        <textarea id="compressed-text" rows="8" class="w-full border rounded p-3 font-mono text-sm bg-gray-50" readonly></textarea>
                    </div>
                </div>
            </div>
        `;
  }

  connectedCallback() {
    this.compressor = null;
    this.btn = this.querySelector("#compress-btn");
    this.statusEl = this.querySelector("#status");
    this.originalText = this.querySelector("#original-text");
    this.compressedText = this.querySelector("#compressed-text");
    this.rateSlider = this.querySelector("#rate");
    this.rateValue = this.querySelector("#rate-value");
    this.modelSelect = this.querySelector("#model-select");
    this.downloadProgress = this.querySelector("#download-progress");
    this.downloadProgressBar = this.querySelector("#download-progress-bar");
    this.downloadProgressValue = this.querySelector("#download-progress-value");

    // Input listeners
    this.originalText.addEventListener("input", (e) => {
      this.querySelector("#original-count").textContent =
        `${e.target.value.length} chars`;
    });

    this.rateSlider.addEventListener("input", ({ target }) => {
      this.rateValue.textContent = Number(target.value).toFixed(2);
    });

    this.modelSelect.addEventListener("change", () => {
      this.compressor = null;
      this.statusEl.textContent = "Ready";
    });

    // Compression action
    this.btn.addEventListener("click", async () => await this.compress());
  }

  setDownloadProgress(value) {
    const isVisible = Number.isFinite(value);
    this.downloadProgress.classList.toggle("opacity-100", isVisible);
    this.downloadProgress.classList.toggle("max-h-12", isVisible);
    this.downloadProgress.classList.toggle("opacity-0", !isVisible);
    this.downloadProgress.classList.toggle("max-h-0", !isVisible);

    if (isVisible) {
      this.downloadProgressValue.textContent =
        this.downloadProgressBar.style.width = `${value.toFixed(1)}%`;
    }
  }

  async initModel() {
    if (!this.compressor) {
      this.btn.disabled = true;
      this.statusEl.textContent = "Loading Model...";
      try {
        const modelKey = this.modelSelect.value;
        this.compressor = await createCompressor(modelKey, {
          onProgress: ({ status, progress }) =>
            status === "progress_total" && this.setDownloadProgress(progress),
        });
        this.setDownloadProgress(null);
        this.statusEl.textContent = "Model Loaded.";
      } catch (err) {
        console.error(err);
        this.setDownloadProgress(null);
        this.statusEl.textContent = "Error loading model. Check console.";
        return false;
      } finally {
        this.btn.disabled = false;
      }
    }
    return true;
  }

  async compress() {
    this.btn.disabled = true;

    const isLoaded = await this.initModel();
    if (!isLoaded) return;

    this.statusEl.textContent = "Compressing...";
    // Yield to event loop to allow render
    await new Promise((r) => setTimeout(r, 10));

    try {
      const textToCompress = this.originalText.value;
      const start = performance.now();
      const result = await this.compressor.compress(textToCompress, {
        rate: parseFloat(this.rateSlider.value),
      });
      const end = performance.now();
      const timeMs = Math.round(end - start);
      this.compressedText.value = result;
      this.querySelector("#compressed-count").textContent =
        `${result.length} chars (Saved ${textToCompress.length - result.length}, took ${timeMs}ms)`;
      this.statusEl.textContent = "Done.";
    } catch (err) {
      console.error(err);
      this.statusEl.textContent = "Error during compression.";
    } finally {
      this.btn.disabled = false;
    }
  }
}

customElements.define("compressor-app", CompressorApp);
