export const PALETTES = {
  aurora: {
    name: "Aurora",
    colors: ["#00D4FF", "#FF6B9D", "#C4FF00", "#FFB700"],
  },
  deepOcean: {
    name: "Deep Ocean",
    colors: ["#00897B", "#FF6F61", "#1A237E", "#26A69A"],
  },
  neon: {
    name: "Neon",
    colors: ["#00FFFF", "#FF00FF", "#00FF00", "#FFFF00"],
  },
};

export const DEFAULT_PALETTE = "aurora";

/**
 * Parse a hex color string into [r, g, b].
 */
export function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
}

/**
 * Linearly interpolate between two [r,g,b] colors.
 */
export function lerpColor(c1, c2, t) {
  return [
    c1[0] + (c2[0] - c1[0]) * t,
    c1[1] + (c2[1] - c1[1]) * t,
    c1[2] + (c2[2] - c1[2]) * t,
  ];
}
