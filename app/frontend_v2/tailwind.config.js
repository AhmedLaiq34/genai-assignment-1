export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#faf8ff', surface: '#ffffff', ink: '#25252e', muted: '#656476',
        line: '#dedbea', wash: '#f4f1ff', track: '#e5e2ed',
        brand: { DEFAULT: '#5140ec', dark: '#3824d7', light: '#dfdaff' },
        success: '#098563', 'success-soft': '#d9fae9', warning: '#b86a0b',
        danger: '#b42338', 'danger-soft': '#fff0f2',
      },
      fontFamily: { sans: ['Arial', 'Helvetica', 'sans-serif'], mono: ['Consolas', 'monospace'] },
      borderRadius: { card: '16px', control: '9px' },
      spacing: { panel: '20px', gutter: '28px' },
      boxShadow: { card: '0 1px 2px rgb(30 20 65 / 6%)', button: '0 3px 6px rgb(54 38 142 / 18%)' },
      maxWidth: { studio: '1600px' },
    },
  },
  plugins: [],
};
