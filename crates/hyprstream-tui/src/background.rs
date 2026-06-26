//! Animated background styles for the empty-window state.
//!
//! `BackgroundState` holds the animation state for the three offered styles:
//! - `Blank`  — solid black, no characters
//! - `Stars`  — sparse braille-dot starfield that slowly twinkles
//! - `Matrix` — katakana/digit columns falling top-to-bottom, green gradient

use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

// ============================================================================
// BackgroundStyle
// ============================================================================

/// Which animated background to show when no windows are open.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum BackgroundStyle {
    #[default]
    Blank,
    Stars,
    Matrix,
}

impl std::fmt::Display for BackgroundStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Blank  => write!(f, "blank"),
            Self::Stars  => write!(f, "stars"),
            Self::Matrix => write!(f, "matrix"),
        }
    }
}

/// All styles in display order (used to build the settings SelectList).
pub const ALL_STYLES: &[BackgroundStyle] = &[
    BackgroundStyle::Blank,
    BackgroundStyle::Stars,
    BackgroundStyle::Matrix,
];

/// Suggested width/height for the preview box inside the settings modal.
pub const PREVIEW_W: u16 = 32;
pub const PREVIEW_H: u16 = 8;

// ============================================================================
// Stars
// ============================================================================

const STAR_CHARS: &[char] = &[
    ' ', ' ', ' ', ' ', ' ',   // mostly empty space — sparse field
    '⠁', '⠂', '⠄', '⠈', '⠐', '⠠', '⡀', '⢀',  // single braille dots
    '·', '✦',
];

struct StarParticle {
    x:      u16,
    y:      u16,
    phase:  u8,  // index into STAR_CHARS
    period: u8,  // ticks between phase increments
}

// ============================================================================
// Matrix
// ============================================================================

const MATRIX_CHARS: &[char] = &[
    'ｦ','ｧ','ｨ','ｩ','ｪ','ｫ','ｬ','ｭ','ｮ','ｯ','ｰ','ｱ','ｲ','ｳ','ｴ','ｵ',
    'ｶ','ｷ','ｸ','ｹ','ｺ','ｻ','ｼ','ｽ','ｾ','ｿ','ﾀ','ﾁ','ﾂ','ﾃ','ﾄ','ﾅ',
    'ﾆ','ﾇ','ﾈ','ﾉ','ﾊ','ﾋ','ﾌ','ﾍ','ﾎ','ﾏ','ﾐ','ﾑ','ﾒ','ﾓ','ﾔ','ﾕ',
    'ﾖ','ﾗ','ﾘ','ﾙ','ﾚ','ﾛ','ﾜ','ﾝ',
    '0','1','2','3','4','5','6','7','8','9',
];

/// Foreground colour for each trail position (0 = head = brightest).
const MATRIX_COLORS: &[Color] = &[
    Color::White,            // head — bright white
    Color::Rgb(0, 220, 40), // trail 1 — bright green
    Color::Rgb(0, 140, 20), // trail 2
    Color::Rgb(0,  70, 10), // trail 3
    Color::Rgb(0,  30,  5), // trail 4 — dim
];

struct MatrixColumn {
    head:    i32, // y-row of the falling head (may be negative before it enters)
    speed:   u8,  // ticks per row advance
    counter: u8,
    trail:   u8,  // length of the trailing glyph streak
    chars:   Vec<char>,
}

// ============================================================================
// LCG pseudo-random helper (no external dep)
// ============================================================================

#[inline]
fn lcg(seed: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ============================================================================
// BackgroundState
// ============================================================================

pub struct BackgroundState {
    pub style:        BackgroundStyle,
    tick:             u64,
    // Dimensions for which stars/matrix were initialised.
    init_w:           u16,
    init_h:           u16,
    init_style:       Option<BackgroundStyle>,
    // Per-style state (only one is populated at a time).
    stars:            Vec<StarParticle>,
    matrix_cols:      Vec<MatrixColumn>,
}

impl BackgroundState {
    pub fn new(style: BackgroundStyle) -> Self {
        Self {
            style,
            tick: 0,
            init_w: 0,
            init_h: 0,
            init_style: None,
            stars: vec![],
            matrix_cols: vec![],
        }
    }

    // ── Initialisation ──────────────────────────────────────────────────────

    fn init_stars(&mut self, w: u16, h: u16) {
        let count = ((w as u32 * h as u32) / 10).clamp(8, 600) as usize;
        self.stars = (0..count)
            .map(|i| {
                let r1 = lcg(i as u64 * 17 + 3);
                let r2 = lcg(r1);
                let r3 = lcg(r2);
                StarParticle {
                    x:      (r1 % w as u64) as u16,
                    y:      (r2 % h as u64) as u16,
                    phase:  (r3 % STAR_CHARS.len() as u64) as u8,
                    period: 4 + (lcg(r3) % 9) as u8,
                }
            })
            .collect();
    }

    fn init_matrix(&mut self, w: u16, h: u16) {
        self.matrix_cols = (0..w as usize)
            .map(|i| {
                let r  = lcg(i as u64 * 31 + 7);
                let r2 = lcg(r);
                let trail = 4 + (r  % 7) as u8;
                let speed = 1 + (r2 % 2) as u8;
                let start = -((lcg(r2) % h as u64) as i32);
                let chars: Vec<char> = (0..trail as usize)
                    .map(|j| MATRIX_CHARS[(i * 5 + j * 11) % MATRIX_CHARS.len()])
                    .collect();
                MatrixColumn { head: start, speed, counter: 0, trail, chars }
            })
            .collect();
    }

    fn needs_reset(&self, w: u16, h: u16) -> bool {
        w != self.init_w || h != self.init_h || self.init_style != Some(self.style)
    }

    // ── Tick ────────────────────────────────────────────────────────────────

    /// Advance the animation by one step.  Call before each render frame.
    pub fn tick(&mut self, w: u16, h: u16) {
        if w == 0 || h == 0 { return; }

        if self.needs_reset(w, h) {
            self.init_w     = w;
            self.init_h     = h;
            self.init_style = Some(self.style);
            self.stars.clear();
            self.matrix_cols.clear();
            match self.style {
                BackgroundStyle::Blank  => {}
                BackgroundStyle::Stars  => self.init_stars(w, h),
                BackgroundStyle::Matrix => self.init_matrix(w, h),
            }
        }

        self.tick = self.tick.wrapping_add(1);

        match self.style {
            BackgroundStyle::Blank => {}

            BackgroundStyle::Stars => {
                for star in &mut self.stars {
                    if self.tick.is_multiple_of(star.period as u64) {
                        star.phase = (star.phase + 1) % STAR_CHARS.len() as u8;
                    }
                }
            }

            BackgroundStyle::Matrix => {
                for (col_idx, col) in self.matrix_cols.iter_mut().enumerate() {
                    col.counter += 1;
                    if col.counter >= col.speed {
                        col.counter = 0;
                        col.head   += 1;
                        // Randomise the head char.
                        let rng = lcg(self.tick ^ col_idx as u64 ^ 0xDEAD_BEEF);
                        col.chars[0] = MATRIX_CHARS[rng as usize % MATRIX_CHARS.len()];
                        col.chars.rotate_right(1);
                        // Wrap when fully below the screen.
                        if col.head - col.trail as i32 >= h as i32 {
                            col.head = -(col.trail as i32);
                        }
                    }
                }
            }
        }
    }

    // ── Render ──────────────────────────────────────────────────────────────

    /// Render the current animation frame into `area`.
    ///
    /// Must be preceded by `tick()` with the same `(area.width, area.height)`;
    /// otherwise the first frame may be blank (harmless).
    pub fn render(&self, frame: &mut Frame, area: Rect) {
        match self.style {
            BackgroundStyle::Blank => {
                frame.render_widget(
                    Paragraph::new("").style(Style::default().bg(Color::Black)),
                    area,
                );
            }

            BackgroundStyle::Stars => {
                let w = area.width as usize;
                let h = area.height as usize;
                let mut grid: Vec<Vec<char>> = (0..h).map(|_| vec![' '; w]).collect();
                for star in &self.stars {
                    let sx = star.x as usize;
                    let sy = star.y as usize;
                    if sy < h && sx < w {
                        grid[sy][sx] = STAR_CHARS[star.phase as usize];
                    }
                }
                let dim = Style::default().fg(Color::DarkGray).bg(Color::Black);
                let lines: Vec<Line> = grid
                    .into_iter()
                    .map(|row| Line::from(Span::styled(row.into_iter().collect::<String>(), dim)))
                    .collect();
                frame.render_widget(Paragraph::new(lines), area);
            }

            BackgroundStyle::Matrix => {
                let w = area.width as usize;
                let h = area.height as usize;
                // 255 = empty cell (black on black)
                let mut grid: Vec<Vec<(char, u8)>> =
                    (0..h).map(|_| vec![(' ', 255u8); w]).collect();

                for (col_idx, col) in self.matrix_cols.iter().enumerate() {
                    if col_idx >= w { break; }
                    for trail_pos in 0..col.trail as i32 {
                        let row = col.head - trail_pos;
                        if row < 0 || row >= h as i32 { continue; }
                        let ch  = col.chars.get(trail_pos as usize).copied().unwrap_or(' ');
                        let lvl = (trail_pos as usize).min(MATRIX_COLORS.len() - 1) as u8;
                        grid[row as usize][col_idx] = (ch, lvl);
                    }
                }

                let lines: Vec<Line> = grid.into_iter().map(|row| {
                    let mut spans:    Vec<Span> = Vec::new();
                    let mut cur_lvl: u8         = 255;
                    let mut text:    String     = String::new();

                    for (ch, lvl) in row {
                        if lvl != cur_lvl {
                            if !text.is_empty() {
                                let fg = color_for_lvl(cur_lvl);
                                spans.push(Span::styled(
                                    std::mem::take(&mut text),
                                    Style::default().fg(fg).bg(Color::Black),
                                ));
                            }
                            cur_lvl = lvl;
                        }
                        text.push(ch);
                    }
                    if !text.is_empty() {
                        spans.push(Span::styled(
                            text,
                            Style::default().fg(color_for_lvl(cur_lvl)).bg(Color::Black),
                        ));
                    }
                    Line::from(spans)
                }).collect();

                frame.render_widget(Paragraph::new(lines), area);
            }
        }
    }

    /// True when the background is animated (i.e. needs periodic redraws).
    pub fn is_animated(&self) -> bool {
        !matches!(self.style, BackgroundStyle::Blank)
    }
}

#[inline]
fn color_for_lvl(lvl: u8) -> Color {
    if lvl as usize >= MATRIX_COLORS.len() {
        Color::Black
    } else {
        MATRIX_COLORS[lvl as usize]
    }
}
