/**
 * OrganicRenderer — Gaussian glow mode (Lenia-style).
 * Soft circles with radial gradient, additive blending for natural overlap glow.
 */

import { BaseRenderer } from "./BaseRenderer.js";
import { GLOW_RADIUS_FACTOR, BACKGROUND_COLOR } from "../config.js";

export class OrganicRenderer extends BaseRenderer {
  constructor(p, options) {
    super(p, options);
    this.glowTexture = null;
    this.textureSize = 0;
  }

  setup() {
    this._buildGlowTexture();
  }

  /**
   * Build an off-screen radial gradient texture (white, alpha falloff).
   * We tint it per-agent at draw time.
   */
  _buildGlowTexture() {
    const p = this.p;
    const size = Math.ceil(Math.max(this.cellWidth, this.cellHeight) * GLOW_RADIUS_FACTOR * 2);
    this.textureSize = Math.max(size, 8);
    const gfx = p.createGraphics(this.textureSize, this.textureSize);

    const cx = this.textureSize / 2;
    const cy = this.textureSize / 2;
    const maxR = this.textureSize / 2;

    gfx.noStroke();
    // Draw concentric circles from center outward with decreasing alpha
    const steps = 32;
    for (let i = steps; i >= 0; i--) {
      const r = (i / steps) * maxR;
      const alpha = Math.pow(1 - i / steps, 1.5) * 255;
      gfx.fill(255, 255, 255, alpha);
      gfx.ellipse(cx, cy, r * 2, r * 2);
    }

    this.glowTexture = gfx;
  }

  resize(panelWidth, panelHeight) {
    super.resize(panelWidth, panelHeight);
    this._buildGlowTexture();
  }

  /**
   * Draw one interpolated frame.
   * @param {Array<{x: number, y: number, color: number[]}>} agents
   * @param {number} offsetX
   * @param {number} offsetY
   */
  drawFrame(agents, offsetX, offsetY) {
    const p = this.p;

    p.push();
    p.translate(offsetX, offsetY);

    // Clip to panel area
    // p5 doesn't have native clip, so we just draw within bounds

    p.blendMode(p.ADD);

    const halfTex = this.textureSize / 2;

    for (const agent of agents) {
      const px = agent.x * this.cellWidth + this.cellWidth / 2;
      const py = agent.y * this.cellHeight + this.cellHeight / 2;

      const [r, g, b] = agent.color;

      p.tint(r, g, b, 200);
      p.image(this.glowTexture, px - halfTex, py - halfTex);
    }

    p.blendMode(p.BLEND);
    p.noTint();
    p.pop();
  }
}
