// Apple Human Interface Guidelines inspired system colors (dark scheme)
// Values chosen to approximate system colors on macOS dark appearance.

export const AppleColors = {
  // Accents
  systemBlue: '#0A84FF',
  systemIndigo: '#5E5CE6',
  systemPurple: '#BF5AF2',
  systemPink: '#FF375F',
  systemRed: '#FF453A',
  systemOrange: '#FF9F0A',
  systemYellow: '#FFD60A',
  systemGreen: '#32D74B',
  systemTeal: '#64D2FF',

  // Grays (dark)
  systemGray: '#8E8E93',
  systemGray2: '#AEAEB2',
  systemGray3: '#C7C7CC',
  systemGray4: '#D1D1D6',
  systemGray5: '#E5E5EA',
  systemGray6: '#1C1C1E',

  // Surfaces
  background: '#000000',
  surface: '#1C1C1E',
  elevated: '#2C2C2E',
};

export type AppleColorKey = keyof typeof AppleColors;
