/**
 * SideBySide — Dual-canvas synchronized layout.
 * Manages two simulation panels sharing one animation clock,
 * plus text overlay with key insight.
 */

import {
  CANVAS_WIDTH,
  GRID_PANEL_HEIGHT,
  METRIC_PANEL_HEIGHT,
  TEXT_PANEL_HEIGHT,
  BACKGROUND_COLOR,
  SUB_FRAMES,
} from "../config.js";
import { PALETTES } from "../palettes.js";
import { interpolateFrame, staticFrame } from "../simulation/Interpolator.js";
import { OrganicRenderer } from "../renderer/OrganicRenderer.js";
import { ParticleRenderer } from "../renderer/ParticleRenderer.js";
import { MetricOverlay } from "./MetricOverlay.js";

const DIVIDER_WIDTH = 2;

export class SideBySide {
  /**
   * @param {Object} p - p5 instance
   * @param {Object} leftData - SimulationData for left panel
   * @param {Object} rightData - SimulationData for right panel
   * @param {Object} pairedMeta - {left_phase, right_phase, sim_seed}
   */
  constructor(p, leftData, rightData, pairedMeta) {
    this.p = p;
    this.leftData = leftData;
    this.rightData = rightData;
    this.pairedMeta = pairedMeta;

    this.panelWidth = (CANVAS_WIDTH - DIVIDER_WIDTH) / 2;
    this.panelHeight = GRID_PANEL_HEIGHT;

    this.leftRenderer = null;
    this.rightRenderer = null;
    this.metricOverlay = null;

    this.currentStep = 0;
    this.subFrame = 0;
    this.textFadeIn = 0;
  }

  setup(renderMode, paletteName) {
    this._createRenderers(renderMode);
    this.metricOverlay = new MetricOverlay(this.p, CANVAS_WIDTH);
  }

  _createRenderers(renderMode) {
    const RendererClass =
      renderMode === "particle" ? ParticleRenderer : OrganicRenderer;

    const leftOpts = {
      gridWidth: this.leftData.meta.grid_width,
      gridHeight: this.leftData.meta.grid_height,
      panelWidth: this.panelWidth,
      panelHeight: this.panelHeight,
    };
    const rightOpts = {
      gridWidth: this.rightData.meta.grid_width,
      gridHeight: this.rightData.meta.grid_height,
      panelWidth: this.panelWidth,
      panelHeight: this.panelHeight,
    };

    this.leftRenderer = new RendererClass(this.p, leftOpts);
    this.rightRenderer = new RendererClass(this.p, rightOpts);
    this.leftRenderer.setup();
    this.rightRenderer.setup();
  }

  switchRenderer(renderMode) {
    this._createRenderers(renderMode);
  }

  /**
   * Advance the animation clock by one sub-frame tick.
   * @returns {{ step: number, done: boolean }}
   */
  tick() {
    this.subFrame++;
    if (this.subFrame >= SUB_FRAMES) {
      this.subFrame = 0;
      this.currentStep++;
    }

    const maxSteps = Math.min(
      this.leftData.frames.length,
      this.rightData.frames.length
    );

    if (this.currentStep >= maxSteps - 1) {
      this.currentStep = maxSteps - 1;
      this.subFrame = 0;
      return { step: this.currentStep, done: true };
    }

    return { step: this.currentStep, done: false };
  }

  /**
   * Reset animation to step 0.
   */
  reset() {
    this.currentStep = 0;
    this.subFrame = 0;
    this.textFadeIn = 0;
  }

  /**
   * Draw the full side-by-side layout for the current tick.
   * @param {string} paletteName
   */
  draw(paletteName) {
    const p = this.p;
    const palette = PALETTES[paletteName];
    const colors = palette.colors;

    // Background
    p.background(BACKGROUND_COLOR);

    // Interpolate left panel
    const leftAgents = this._interpolateSide(
      this.leftData,
      colors,
      this.leftData.meta.grid_width,
      this.leftData.meta.grid_height
    );
    // Interpolate right panel
    const rightAgents = this._interpolateSide(
      this.rightData,
      colors,
      this.rightData.meta.grid_width,
      this.rightData.meta.grid_height
    );

    // Draw panels
    this.leftRenderer.drawFrame(leftAgents, 0, 0);
    this.rightRenderer.drawFrame(
      rightAgents,
      this.panelWidth + DIVIDER_WIDTH,
      0
    );

    // Divider line
    p.stroke(60);
    p.strokeWeight(DIVIDER_WIDTH);
    p.line(this.panelWidth, 0, this.panelWidth, this.panelHeight);

    // Panel labels
    p.noStroke();
    p.fill(255, 255, 255, 180);
    p.textSize(14);
    p.textAlign(p.CENTER, p.TOP);
    const leftLabel = this._phaseLabel(this.pairedMeta.left_phase);
    const rightLabel = this._phaseLabel(this.pairedMeta.right_phase);
    p.text(leftLabel, this.panelWidth / 2, 8);
    p.text(rightLabel, this.panelWidth + DIVIDER_WIDTH + this.panelWidth / 2, 8);

    // Subtitle under labels
    p.textSize(10);
    p.fill(255, 255, 255, 100);
    p.text("(sees neighbors)", this.panelWidth / 2, 26);
    p.text("(sees clock)", this.panelWidth + DIVIDER_WIDTH + this.panelWidth / 2, 26);

    // Metric overlay
    const leftMI = this.leftData.metrics?.neighbor_mutual_information;
    const rightMI = this.rightData.metrics?.neighbor_mutual_information;

    if (leftMI) {
      this.metricOverlay.draw({
        leftMI,
        rightMI,
        leftColor: [0, 180, 255],
        rightColor: [255, 100, 100],
        currentStep: this.currentStep,
        leftLabel,
        rightLabel,
      });
    }

    // Text panel with fade-in
    this._drawTextPanel();
  }

  _interpolateSide(data, colors, gridW, gridH) {
    const step = this.currentStep;
    const frames = data.frames;

    if (step >= frames.length - 1) {
      return staticFrame(frames[frames.length - 1].agents, colors);
    }

    if (this.subFrame === 0 && step === 0) {
      return staticFrame(frames[0].agents, colors);
    }

    return interpolateFrame(
      frames[step].agents,
      frames[step + 1].agents,
      gridW,
      gridH,
      colors,
      this.subFrame
    );
  }

  _phaseLabel(phase) {
    const labels = {
      phase_1: "Phase 1",
      phase_2: "Phase 2",
      control: "Control",
      random_walk: "Random Walk",
    };
    return labels[phase] || phase;
  }

  _drawTextPanel() {
    const p = this.p;
    const y = GRID_PANEL_HEIGHT + METRIC_PANEL_HEIGHT;

    // Fade in after step 30
    if (this.currentStep > 30) {
      this.textFadeIn = Math.min(this.textFadeIn + 0.02, 1);
    }

    p.noStroke();
    p.fill(10, 10, 18);
    p.rect(0, y, CANVAS_WIDTH, TEXT_PANEL_HEIGHT);

    if (this.textFadeIn > 0) {
      const alpha = this.textFadeIn * 200;
      p.fill(255, 255, 255, alpha);
      p.textSize(16);
      p.textAlign(p.CENTER, p.CENTER);
      p.text(
        "Same starting conditions. Only difference: what they see.",
        CANVAS_WIDTH / 2,
        y + TEXT_PANEL_HEIGHT / 2
      );
    }
  }
}
