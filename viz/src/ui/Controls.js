/**
 * Controls — Play/pause, speed, palette, render mode, and record GIF.
 * Creates DOM elements and exposes reactive state.
 */

import { PALETTES, DEFAULT_PALETTE } from "../palettes.js";
import { DEFAULT_FPS } from "../config.js";

export class Controls {
  constructor(containerEl) {
    this.container = containerEl;
    this.playing = true;
    this.speed = 1.0;
    this.paletteName = DEFAULT_PALETTE;
    this.renderMode = "organic";
    this.onRecord = null;

    this._onChange = null;
    this._keyHandler = null;
    this._build();
    this._bindKeyboard();
  }

  /**
   * Register a callback fired on any control change.
   * @param {Function} fn
   */
  onChange(fn) {
    this._onChange = fn;
  }

  _notify() {
    if (this._onChange) this._onChange(this.getState());
  }

  getState() {
    return {
      playing: this.playing,
      speed: this.speed,
      paletteName: this.paletteName,
      renderMode: this.renderMode,
    };
  }

  getEffectiveFPS() {
    return DEFAULT_FPS * this.speed;
  }

  _build() {
    const bar = document.createElement("div");
    bar.className = "controls-bar";

    // Play/Pause button
    this.playBtn = document.createElement("button");
    this.playBtn.textContent = "Pause";
    this.playBtn.className = "ctrl-btn";
    this.playBtn.addEventListener("click", () => this.togglePlay());
    bar.appendChild(this.playBtn);

    // Speed slider
    const speedLabel = document.createElement("label");
    speedLabel.className = "ctrl-label";
    speedLabel.textContent = "Speed: 1.0x";
    const speedSlider = document.createElement("input");
    speedSlider.type = "range";
    speedSlider.min = "0.5";
    speedSlider.max = "4";
    speedSlider.step = "0.5";
    speedSlider.value = "1";
    speedSlider.className = "ctrl-slider";
    speedSlider.addEventListener("input", () => {
      this.speed = parseFloat(speedSlider.value);
      speedLabel.textContent = `Speed: ${this.speed.toFixed(1)}x`;
      this._notify();
    });
    bar.appendChild(speedLabel);
    bar.appendChild(speedSlider);

    // Palette selector
    const paletteSelect = document.createElement("select");
    paletteSelect.className = "ctrl-select";
    for (const [key, pal] of Object.entries(PALETTES)) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = pal.name;
      if (key === DEFAULT_PALETTE) opt.selected = true;
      paletteSelect.appendChild(opt);
    }
    paletteSelect.addEventListener("change", () => {
      this.paletteName = paletteSelect.value;
      this._notify();
    });
    const palLabel = document.createElement("label");
    palLabel.className = "ctrl-label";
    palLabel.textContent = "Palette:";
    bar.appendChild(palLabel);
    bar.appendChild(paletteSelect);

    // Render mode toggle
    const modeBtn = document.createElement("button");
    modeBtn.textContent = "Mode: Organic";
    modeBtn.className = "ctrl-btn";
    modeBtn.addEventListener("click", () => {
      this.renderMode = this.renderMode === "organic" ? "particle" : "organic";
      modeBtn.textContent =
        this.renderMode === "organic" ? "Mode: Organic" : "Mode: Particle";
      this._notify();
    });
    bar.appendChild(modeBtn);

    // Record GIF button
    const recBtn = document.createElement("button");
    recBtn.textContent = "Record GIF";
    recBtn.className = "ctrl-btn ctrl-rec";
    recBtn.addEventListener("click", () => {
      if (this.onRecord) this.onRecord();
    });
    bar.appendChild(recBtn);

    this.container.appendChild(bar);
  }

  togglePlay() {
    this.playing = !this.playing;
    this.playBtn.textContent = this.playing ? "Pause" : "Play";
    this._notify();
  }

  _bindKeyboard() {
    this._keyHandler = (e) => {
      if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        this.togglePlay();
      }
    };
    document.addEventListener("keydown", this._keyHandler);
  }

  destroy() {
    if (this._keyHandler) {
      document.removeEventListener("keydown", this._keyHandler);
      this._keyHandler = null;
    }
  }
}
