/**
 * DataLoader — Fetch and parse exported JSON simulation data.
 * Handles both single-simulation and paired (side-by-side) formats.
 */

/**
 * @typedef {Object} SimulationData
 * @property {Object} meta
 * @property {Array<{step: number, agents: Array<[number, number, number]>}>} frames
 * @property {Object} metrics
 */

/**
 * @typedef {Object} PairedData
 * @property {SimulationData} left
 * @property {SimulationData} right
 * @property {Object} meta
 */

/**
 * Load simulation JSON from a URL or file path.
 * @param {string} url
 * @returns {Promise<SimulationData | PairedData>}
 */
export async function loadSimulationData(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status} ${response.statusText}`);
  }
  const data = await response.json();
  return data;
}

/**
 * Determine if loaded data is a paired (side-by-side) format.
 * @param {Object} data
 * @returns {boolean}
 */
export function isPairedData(data) {
  return data.left !== undefined && data.right !== undefined;
}

/**
 * Extract a single simulation's data, whether from single or paired format.
 * @param {Object} data
 * @param {"left" | "right" | null} side - null for single format
 * @returns {SimulationData}
 */
export function getSimulationSide(data, side) {
  if (side === null) {
    return data;
  }
  return data[side];
}
