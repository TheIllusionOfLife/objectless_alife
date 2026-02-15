/**
 * MetricOverlay — MI time-series line chart below the grid panels.
 * In side-by-side mode, overlays both series with phase-colored lines.
 */

import { METRIC_PANEL_HEIGHT, GRID_PANEL_HEIGHT } from "../config.js";

const MARGIN = { top: 10, right: 20, bottom: 25, left: 50 };
const LABEL_COLOR = [180, 180, 180];
const MARKER_COLOR = [255, 255, 255, 120];

export class MetricOverlay {
  /**
   * @param {Object} p - p5 instance
   * @param {number} width - overlay width in pixels
   */
  constructor(p, width) {
    this.p = p;
    this.width = width;
    this.height = METRIC_PANEL_HEIGHT;
    this.offsetY = GRID_PANEL_HEIGHT;
  }

  /**
   * Draw the MI time-series chart.
   * @param {Object} options
   * @param {number[]} options.leftMI - MI values array (or null)
   * @param {number[]} options.rightMI - MI values array (or null, for side-by-side)
   * @param {number[]} options.leftColor - [r, g, b] for left series
   * @param {number[]} options.rightColor - [r, g, b] for right series (or null)
   * @param {number} options.currentStep - current simulation step
   * @param {string} [options.leftLabel]
   * @param {string} [options.rightLabel]
   */
  draw({
    leftMI,
    rightMI = null,
    leftColor,
    rightColor = null,
    currentStep,
    leftLabel = "Phase 2",
    rightLabel = "Control",
  }) {
    const p = this.p;

    p.push();
    p.translate(0, this.offsetY);

    // Background
    p.noStroke();
    p.fill(10, 10, 18);
    p.rect(0, 0, this.width, this.height);

    const plotX = MARGIN.left;
    const plotY = MARGIN.top;
    const plotW = this.width - MARGIN.left - MARGIN.right;
    const plotH = this.height - MARGIN.top - MARGIN.bottom;

    const primaryMI = leftMI || rightMI;
    if (!primaryMI || primaryMI.length === 0) {
      p.pop();
      return;
    }

    // Compute y-axis range
    let allValues = leftMI ? [...leftMI] : [];
    if (rightMI) allValues = allValues.concat(rightMI);
    const yMax = Math.max(...allValues, 0.01);
    const yMin = 0;
    const totalSteps = primaryMI.length;
    const stepDivisor = totalSteps > 1 ? totalSteps - 1 : 1;

    // Axis labels
    p.textSize(10);
    p.textAlign(p.CENTER, p.TOP);
    p.fill(...LABEL_COLOR);
    p.noStroke();
    p.text("Step", plotX + plotW / 2, this.height - 14);

    p.push();
    p.translate(12, plotY + plotH / 2);
    p.rotate(-p.HALF_PI);
    p.textAlign(p.CENTER, p.BOTTOM);
    p.text("MI", 0, 0);
    p.pop();

    // Axes
    p.stroke(...LABEL_COLOR, 60);
    p.strokeWeight(1);
    p.line(plotX, plotY, plotX, plotY + plotH);
    p.line(plotX, plotY + plotH, plotX + plotW, plotY + plotH);

    // Tick labels
    p.noStroke();
    p.fill(...LABEL_COLOR);
    p.textAlign(p.RIGHT, p.CENTER);
    p.textSize(9);
    p.text(yMax.toFixed(2), plotX - 4, plotY);
    p.text("0", plotX - 4, plotY + plotH);
    p.textAlign(p.CENTER, p.TOP);
    p.text("0", plotX, plotY + plotH + 2);
    p.text(String(totalSteps - 1), plotX + plotW, plotY + plotH + 2);

    // Draw series
    const drawSeries = (values, color) => {
      p.noFill();
      p.stroke(color[0], color[1], color[2], 220);
      p.strokeWeight(1.5);
      p.beginShape();
      for (let i = 0; i < values.length; i++) {
        const sx = plotX + (i / stepDivisor) * plotW;
        const sy = plotY + plotH - ((values[i] - yMin) / (yMax - yMin)) * plotH;
        p.vertex(sx, sy);
      }
      p.endShape();
    };

    if (leftMI) drawSeries(leftMI, leftColor);
    if (rightMI && rightColor) {
      drawSeries(rightMI, rightColor);
    }

    // Current step marker
    if (currentStep >= 0 && currentStep < totalSteps) {
      const mx = plotX + (currentStep / stepDivisor) * plotW;
      p.stroke(...MARKER_COLOR);
      p.strokeWeight(1);
      p.line(mx, plotY, mx, plotY + plotH);
    }

    // Legend
    if (rightMI) {
      p.noStroke();
      p.textSize(9);
      p.textAlign(p.LEFT, p.CENTER);
      const legendX = plotX + plotW - 120;
      const legendY = plotY + 8;

      p.fill(leftColor[0], leftColor[1], leftColor[2]);
      p.rect(legendX, legendY, 10, 3);
      p.fill(...LABEL_COLOR);
      p.text(leftLabel, legendX + 14, legendY + 1);

      p.fill(rightColor[0], rightColor[1], rightColor[2]);
      p.rect(legendX, legendY + 12, 10, 3);
      p.fill(...LABEL_COLOR);
      p.text(rightLabel, legendX + 14, legendY + 13);
    }

    p.pop();
  }
}
