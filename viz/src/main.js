/**
 * Main entry point — URL param routing, p5 instance setup.
 *
 * URL params:
 *   ?data=path/to/simulation.json   Single simulation view
 *   ?paired=path/to/paired.json     Side-by-side comparison
 *   ?palette=aurora|deepOcean|neon
 *   ?mode=organic|particle
 */

import p5 from "p5";
import {
  CANVAS_WIDTH,
  CANVAS_HEIGHT,
  GRID_PANEL_HEIGHT,
  SUB_FRAMES,
  BACKGROUND_COLOR,
} from "./config.js";
import { PALETTES, DEFAULT_PALETTE } from "./palettes.js";
import { loadSimulationData, isPairedData } from "./simulation/DataLoader.js";
import { interpolateFrame, staticFrame } from "./simulation/Interpolator.js";
import { OrganicRenderer } from "./renderer/OrganicRenderer.js";
import { ParticleRenderer } from "./renderer/ParticleRenderer.js";
import { Controls } from "./ui/Controls.js";
import { SideBySide } from "./ui/SideBySide.js";
import { MetricOverlay } from "./ui/MetricOverlay.js";
import { GifRecorder } from "./recording/GifRecorder.js";

// Parse URL params
const params = new URLSearchParams(window.location.search);
const dataUrl = params.get("data");
const pairedUrl = params.get("paired");
const rawPalette = params.get("palette");
const initialPalette = rawPalette && PALETTES[rawPalette] ? rawPalette : DEFAULT_PALETTE;
const initialMode = params.get("mode") || "organic";

async function main() {
  if (!dataUrl && !pairedUrl) {
    document.getElementById("app").innerHTML =
      '<div class="message">' +
      "<h2>Objectless ALife Visualization</h2>" +
      "<p>Provide simulation data via URL parameters:</p>" +
      "<ul>" +
      '<li><code>?data=path/to/simulation.json</code> — Single view</li>' +
      '<li><code>?paired=path/to/paired.json</code> — Side-by-side comparison</li>' +
      "</ul>" +
      "<p>Optional: <code>&palette=aurora</code>, <code>&mode=organic</code></p>" +
      "</div>";
    return;
  }

  const url = pairedUrl || dataUrl;
  let rawData;
  try {
    rawData = await loadSimulationData(url);
  } catch (err) {
    const errorDiv = document.createElement("div");
    errorDiv.className = "message error";
    errorDiv.textContent = `Failed to load data: ${err.message}`;
    const app = document.getElementById("app");
    app.innerHTML = "";
    app.appendChild(errorDiv);
    return;
  }

  const paired = isPairedData(rawData);

  // Set up controls
  const controlsEl = document.getElementById("controls");
  const controls = new Controls(controlsEl);
  controls.paletteName = initialPalette;
  controls.renderMode = initialMode;

  const gifRecorder = new GifRecorder();

  if (paired) {
    setupPairedMode(rawData, controls, gifRecorder);
  } else {
    setupSingleMode(rawData, controls, gifRecorder);
  }
}

function setupSingleMode(data, controls, gifRecorder) {
  const sketch = (p) => {
    let renderer;
    let metricOverlay;
    let currentStep = 0;
    let subFrame = 0;

    const gridW = data.meta.grid_width;
    const gridH = data.meta.grid_height;
    const totalFrames = data.frames.length;

    function createRenderer() {
      const RendererClass =
        controls.renderMode === "particle" ? ParticleRenderer : OrganicRenderer;
      renderer = new RendererClass(p, {
        gridWidth: gridW,
        gridHeight: gridH,
        panelWidth: CANVAS_WIDTH,
        panelHeight: GRID_PANEL_HEIGHT,
      });
      renderer.setup();
    }

    p.setup = () => {
      p.createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
      p.frameRate(controls.getEffectiveFPS() * SUB_FRAMES);
      createRenderer();
      metricOverlay = new MetricOverlay(p, CANVAS_WIDTH);
    };

    p.draw = () => {
      p.frameRate(controls.getEffectiveFPS() * SUB_FRAMES);
      p.background(BACKGROUND_COLOR);

      const palette = PALETTES[controls.paletteName];
      const colors = palette.colors;

      // Interpolate agents
      let agents;
      if (currentStep >= totalFrames - 1) {
        agents = staticFrame(data.frames[totalFrames - 1].agents, colors);
      } else if (subFrame === 0 && currentStep === 0) {
        agents = staticFrame(data.frames[0].agents, colors);
      } else {
        agents = interpolateFrame(
          data.frames[currentStep].agents,
          data.frames[currentStep + 1].agents,
          gridW,
          gridH,
          colors,
          subFrame
        );
      }

      renderer.drawFrame(agents, 0, 0);

      // Metric overlay
      const mi = data.metrics?.neighbor_mutual_information;
      if (mi) {
        metricOverlay.draw({
          leftMI: mi,
          leftColor: [0, 212, 255],
          currentStep,
        });
      }

      // Step info
      p.noStroke();
      p.fill(255, 255, 255, 150);
      p.textSize(12);
      p.textAlign(p.LEFT, p.BOTTOM);
      p.text(
        `Step: ${currentStep} / ${totalFrames - 1}  |  ${data.meta.rule_id || ""}`,
        10,
        CANVAS_HEIGHT - 10
      );

      // GIF recording
      if (gifRecorder.recording) {
        gifRecorder.addFrame(p.canvas, currentStep);
        // Red dot indicator
        p.fill(255, 0, 0);
        p.noStroke();
        p.ellipse(CANVAS_WIDTH - 20, 20, 12, 12);
      }

      // Advance clock
      if (controls.playing && currentStep < totalFrames - 1) {
        subFrame++;
        if (subFrame >= SUB_FRAMES) {
          subFrame = 0;
          currentStep++;
        }
      }
    };

    controls.onChange((state) => {
      if (
        (state.renderMode === "particle" && renderer instanceof OrganicRenderer) ||
        (state.renderMode === "organic" && renderer instanceof ParticleRenderer)
      ) {
        createRenderer();
      }
    });

    controls.onRecord = () => {
      currentStep = 0;
      subFrame = 0;
      controls.playing = true;
      gifRecorder.start();
    };
  };

  new p5(sketch, document.getElementById("app"));
}

function setupPairedMode(rawData, controls, gifRecorder) {
  const leftData = rawData.left;
  const rightData = rawData.right;
  const pairedMeta = rawData.meta;

  const sketch = (p) => {
    let sideBySide;

    p.setup = () => {
      p.createCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
      p.frameRate(controls.getEffectiveFPS() * SUB_FRAMES);
      sideBySide = new SideBySide(p, leftData, rightData, pairedMeta);
      sideBySide.setup(controls.renderMode);
    };

    p.draw = () => {
      p.frameRate(controls.getEffectiveFPS() * SUB_FRAMES);

      if (controls.playing) {
        const { done } = sideBySide.tick();
        if (done) {
          controls.playing = false;
        }
      }

      sideBySide.draw(controls.paletteName);

      // Step info
      p.noStroke();
      p.fill(255, 255, 255, 100);
      p.textSize(10);
      p.textAlign(p.RIGHT, p.BOTTOM);
      p.text(`Step: ${sideBySide.currentStep}`, CANVAS_WIDTH - 10, GRID_PANEL_HEIGHT - 5);

      // GIF recording
      if (gifRecorder.recording) {
        const cont = gifRecorder.addFrame(p.canvas, sideBySide.currentStep);
        if (!cont) {
          controls.playing = false;
        }
        p.fill(255, 0, 0);
        p.noStroke();
        p.ellipse(CANVAS_WIDTH - 20, 20, 12, 12);
      }
    };

    controls.onChange((state) => {
      if (
        (state.renderMode === "particle" &&
          sideBySide.leftRenderer instanceof OrganicRenderer) ||
        (state.renderMode === "organic" &&
          sideBySide.leftRenderer instanceof ParticleRenderer)
      ) {
        sideBySide.switchRenderer(state.renderMode);
      }
    });

    controls.onRecord = () => {
      sideBySide.reset();
      controls.playing = true;
      gifRecorder.start();
    };
  };

  new p5(sketch, document.getElementById("app"));
}

main().catch((err) => console.error("Fatal:", err));
