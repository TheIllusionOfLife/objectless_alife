/**
 * BaseRenderer — Shared interface for rendering modes.
 * Subclasses must implement setup(), drawFrame(), and resize().
 */
export class BaseRenderer {
  /**
   * @param {Object} p - p5 instance
   * @param {Object} options
   * @param {number} options.gridWidth
   * @param {number} options.gridHeight
   * @param {number} options.panelWidth - pixel width of render area
   * @param {number} options.panelHeight - pixel height of render area
   */
  constructor(p, options) {
    this.p = p;
    this.gridWidth = options.gridWidth;
    this.gridHeight = options.gridHeight;
    this.panelWidth = options.panelWidth;
    this.panelHeight = options.panelHeight;
    this.cellWidth = this.panelWidth / this.gridWidth;
    this.cellHeight = this.panelHeight / this.gridHeight;
  }

  /** Called once after p5 setup. Override in subclass. */
  setup() {}

  /**
   * Draw one interpolated frame of agents.
   * @param {Array<{x: number, y: number, color: number[], prevState: number, nextState: number}>} agents
   * @param {number} offsetX - pixel offset for this panel
   * @param {number} offsetY - pixel offset for this panel
   */
  drawFrame(agents, offsetX, offsetY) {
    throw new Error("drawFrame() must be implemented by subclass");
  }

  /**
   * Handle resize.
   * @param {number} panelWidth
   * @param {number} panelHeight
   */
  resize(panelWidth, panelHeight) {
    this.panelWidth = panelWidth;
    this.panelHeight = panelHeight;
    this.cellWidth = this.panelWidth / this.gridWidth;
    this.cellHeight = this.panelHeight / this.gridHeight;
  }
}
