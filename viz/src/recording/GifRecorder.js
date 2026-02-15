/**
 * GifRecorder — gif.js wrapper for client-side GIF encoding.
 * Records canvas frames and produces a downloadable GIF.
 */

import { GIF_WIDTH, GIF_HEIGHT, DEFAULT_FPS, GIF_LOOP_END_STEP, GIF_QUALITY } from "../config.js";

export class GifRecorder {
  constructor() {
    this.gif = null;
    this.recording = false;
    this.frameCount = 0;
    this._tempCanvas = null;
  }

  /**
   * Start recording. Creates a new GIF encoder instance.
   * gif.js must be loaded via script tag or import.
   */
  start() {
    if (this.recording) {
      console.warn("GIF recording already in progress.");
      return;
    }

    if (typeof GIF === "undefined") {
      console.error("gif.js not loaded. Include gif.js via script tag.");
      return;
    }

    this.gif = new GIF({
      workers: 2,
      quality: GIF_QUALITY,
      width: GIF_WIDTH,
      height: GIF_HEIGHT,
      workerScript: "/gif.worker.js",
      repeat: 0, // loop forever
    });

    this.recording = true;
    this.frameCount = 0;

    this.gif.on("finished", (blob) => {
      this._download(blob);
      this.recording = false;
    });

    console.log("GIF recording started");
  }

  /**
   * Add the current canvas frame to the GIF.
   * Call this each rendered frame during recording.
   * @param {HTMLCanvasElement} canvas
   * @param {number} currentStep
   * @returns {boolean} true if recording should continue
   */
  addFrame(canvas, currentStep) {
    if (!this.recording || !this.gif) return false;

    // Reuse a single off-screen canvas for GIF frames
    if (!this._tempCanvas) {
      this._tempCanvas = document.createElement("canvas");
      this._tempCanvas.width = GIF_WIDTH;
      this._tempCanvas.height = GIF_HEIGHT;
    }
    const ctx = this._tempCanvas.getContext("2d");
    ctx.drawImage(canvas, 0, 0, GIF_WIDTH, GIF_HEIGHT);

    const delay = Math.round(1000 / DEFAULT_FPS);
    this.gif.addFrame(this._tempCanvas, { copy: true, delay });
    this.frameCount++;

    // Stop after reaching the loop end step
    if (currentStep >= GIF_LOOP_END_STEP) {
      this.stop();
      return false;
    }

    return true;
  }

  /**
   * Stop recording and render the GIF.
   */
  stop() {
    if (!this.recording || !this.gif) return;
    console.log(`GIF recording stopped. Rendering ${this.frameCount} frames...`);
    this.gif.render();
  }

  /**
   * Trigger a browser download of the rendered GIF blob.
   * @param {Blob} blob
   */
  _download(blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `objectless-alife-${Date.now()}.gif`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log("GIF download triggered");
  }
}
