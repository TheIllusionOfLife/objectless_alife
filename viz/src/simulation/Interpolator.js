/**
 * Interpolator — Sub-frame lerp with toroidal wrap handling.
 *
 * Given two consecutive frames of agent positions+states, produces
 * intermediate positions for smooth animation.
 */

import { SUB_FRAMES } from "../config.js";
import { hexToRgb, lerpColor } from "../palettes.js";

/**
 * Compute the shortest-path delta on a toroidal axis.
 * If |raw delta| > size/2, wrap through the boundary.
 * @param {number} from
 * @param {number} to
 * @param {number} size - grid dimension on this axis
 * @returns {number} signed delta (shortest path)
 */
function toroidalDelta(from, to, size) {
  let d = to - from;
  if (d > size / 2) d -= size;
  if (d < -size / 2) d += size;
  return d;
}

/**
 * Wrap a coordinate into [0, size) range.
 * @param {number} v
 * @param {number} size
 * @returns {number}
 */
function wrapCoord(v, size) {
  return ((v % size) + size) % size;
}

/**
 * Generate interpolated agent data between two frames.
 *
 * @param {Array<[number, number, number]>} agentsPrev - Previous frame agents [x, y, state]
 * @param {Array<[number, number, number]>} agentsNext - Next frame agents [x, y, state]
 * @param {number} gridWidth
 * @param {number} gridHeight
 * @param {string[]} paletteColors - Hex color strings for each state
 * @param {number} subFrameIndex - 0..SUB_FRAMES-1
 * @returns {Array<{x: number, y: number, color: number[], prevState: number, nextState: number}>}
 */
export function interpolateFrame(
  agentsPrev,
  agentsNext,
  gridWidth,
  gridHeight,
  paletteColors,
  subFrameIndex
) {
  const t = subFrameIndex / SUB_FRAMES;
  const result = [];

  for (let i = 0; i < agentsPrev.length; i++) {
    const [px, py, ps] = agentsPrev[i];
    const [nx, ny, ns] = agentsNext[i];

    const dx = toroidalDelta(px, nx, gridWidth);
    const dy = toroidalDelta(py, ny, gridHeight);

    const x = wrapCoord(px + dx * t, gridWidth);
    const y = wrapCoord(py + dy * t, gridHeight);

    // Cross-fade color if state changes
    const prevColor = hexToRgb(paletteColors[ps]);
    const nextColor = hexToRgb(paletteColors[ns]);
    const color = ps === ns ? prevColor : lerpColor(prevColor, nextColor, t);

    result.push({ x, y, color, prevState: ps, nextState: ns });
  }

  return result;
}

/**
 * For the very first frame (no previous), just return agents at their positions.
 * @param {Array<[number, number, number]>} agents
 * @param {string[]} paletteColors
 * @returns {Array<{x: number, y: number, color: number[], prevState: number, nextState: number}>}
 */
export function staticFrame(agents, paletteColors) {
  return agents.map(([x, y, s]) => ({
    x,
    y,
    color: hexToRgb(paletteColors[s]),
    prevState: s,
    nextState: s,
  }));
}
