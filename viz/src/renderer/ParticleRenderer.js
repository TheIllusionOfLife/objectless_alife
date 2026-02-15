/**
 * ParticleRenderer — Trails + motion blur (Particle Life-style).
 * Small bright dots with trailing polyline (last N positions, fading alpha).
 * State changes produce a brief white flash.
 */

import { BaseRenderer } from "./BaseRenderer.js";
import { TRAIL_LENGTH } from "../config.js";

export class ParticleRenderer extends BaseRenderer {
  constructor(p, options) {
    super(p, options);
    // trails[i] = array of {x, y, color} for agent i, most recent last
    this.trails = [];
    this.flashTimers = [];
  }

  setup() {
    this.trails = [];
    this.flashTimers = [];
  }

  resize(panelWidth, panelHeight) {
    super.resize(panelWidth, panelHeight);
  }

  /**
   * Draw one interpolated frame with trails.
   * @param {Array<{x: number, y: number, color: number[], prevState: number, nextState: number}>} agents
   * @param {number} offsetX
   * @param {number} offsetY
   */
  drawFrame(agents, offsetX, offsetY) {
    const p = this.p;

    // Initialize trails if needed
    if (this.trails.length !== agents.length) {
      this.trails = agents.map(() => []);
      this.flashTimers = agents.map(() => 0);
    }

    p.push();
    p.translate(offsetX, offsetY);

    // Draw faint grid lines for context
    p.stroke(255, 255, 255, 15);
    p.strokeWeight(0.5);
    for (let gx = 0; gx <= this.gridWidth; gx++) {
      const px = gx * this.cellWidth;
      p.line(px, 0, px, this.panelHeight);
    }
    for (let gy = 0; gy <= this.gridHeight; gy++) {
      const py = gy * this.cellHeight;
      p.line(0, py, this.panelWidth, py);
    }

    for (let i = 0; i < agents.length; i++) {
      const agent = agents[i];
      const px = agent.x * this.cellWidth + this.cellWidth / 2;
      const py = agent.y * this.cellHeight + this.cellHeight / 2;

      // Detect state change — trigger flash
      if (agent.prevState !== agent.nextState) {
        this.flashTimers[i] = 4; // flash for 4 sub-frames
      }

      // Record trail position
      this.trails[i].push({ x: px, y: py, color: [...agent.color] });
      if (this.trails[i].length > TRAIL_LENGTH) {
        this.trails[i].shift();
      }

      // Draw trail polyline with fading alpha
      const trail = this.trails[i];
      p.noFill();
      for (let j = 1; j < trail.length; j++) {
        const alpha = (j / trail.length) * 120;
        const tc = trail[j].color;
        p.stroke(tc[0], tc[1], tc[2], alpha);
        p.strokeWeight(2);

        // Only draw line segment if points are close (skip toroidal wraps)
        const dx = Math.abs(trail[j].x - trail[j - 1].x);
        const dy = Math.abs(trail[j].y - trail[j - 1].y);
        if (dx < this.panelWidth / 2 && dy < this.panelHeight / 2) {
          p.line(trail[j - 1].x, trail[j - 1].y, trail[j].x, trail[j].y);
        }
      }

      // Draw agent dot
      p.noStroke();
      if (this.flashTimers[i] > 0) {
        // White flash
        const flashAlpha = (this.flashTimers[i] / 4) * 255;
        p.fill(255, 255, 255, flashAlpha);
        p.ellipse(px, py, this.cellWidth * 0.8, this.cellHeight * 0.8);
        this.flashTimers[i]--;
      } else {
        const [r, g, b] = agent.color;
        p.fill(r, g, b, 240);
        p.ellipse(px, py, this.cellWidth * 0.6, this.cellHeight * 0.6);
      }
    }

    p.pop();
  }
}
