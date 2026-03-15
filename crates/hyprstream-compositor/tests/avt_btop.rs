//! Verify avt correctly handles btop's escape sequences without leaking
//! unparsed control codes into visible cell content.

#[test]
fn btop_sequences_render_correctly() {
    // Simulate btop's startup + rendering sequences
    let seq = concat!(
        "\x1b[?1049h",                              // enter alt screen
        "\x1b[?25l",                                // hide cursor
        "\x1b[H\x1b[2J",                            // home + clear
        "\x1b[38;2;255;100;0m\x1b[48;2;20;20;40m", // truecolor fg+bg
        "btop header text here",
        "\x1b[0m",
        "\x1b[2;1H",                                // move to row 2
        "block: \u{2588}\u{2593}\u{2592}\u{2591} done",
        "\x1b[3;1H",
        "box: \u{250c}\u{2500}\u{252c}\u{2500}\u{2510}",
        "\x1b[?2026h",                              // sync output — should be silently ignored
        "\x1b[4;1H",
        "after sync mode: visible?",
        "\x1b[?1l\x1b>",                            // decrst mode 1 + DECKPNM
        "\x1b[5;1H",
        "after deckpnm",
    );

    let mut vt = avt::Vt::builder().size(80, 24).build();
    vt.feed_str(seq);

    let rows: Vec<String> = vt
        .view()
        .take(8)
        .map(|line| {
            line.cells()
                .iter()
                .filter(|c| c.width() > 0)
                .map(|c| c.char())
                .collect::<String>()
        })
        .collect();

    for (i, row) in rows.iter().enumerate() {
        eprintln!("row[{i}]: {:?}", row.trim_end());
    }

    // Row 0: header text (written via CSI H + truecolor SGR)
    assert!(
        rows[0].contains("btop header"),
        "row[0] missing header: {:?}",
        rows[0]
    );

    // Row 1: block chars (U+2588 etc.) must appear, not as escaped sequences
    assert!(
        rows[1].contains('\u{2588}'),
        "row[1] missing block char: {:?}",
        rows[1]
    );

    // Row 2: box-drawing chars
    assert!(
        rows[2].contains('\u{250c}'),
        "row[2] missing box char: {:?}",
        rows[2]
    );

    // Row 3: text after ?2026h — if sync mode leaked, this would be missing or garbled
    assert!(
        rows[3].contains("after sync mode"),
        "row[3] missing (did ?2026h leak?): {:?}",
        rows[3]
    );

    // Row 4: text after ?1l + ESC> (DECKPNM)
    assert!(
        rows[4].contains("after deckpnm"),
        "row[4] missing (did \\x1b> leak?): {:?}",
        rows[4]
    );

    // No row should contain raw ESC characters or CSI-like text
    for (i, row) in rows.iter().enumerate() {
        assert!(
            !row.contains('\x1b'),
            "row[{i}] contains raw ESC: {:?}",
            row
        );
        // Check for CSI leak patterns like "?2026h" or "[2J"
        assert!(
            !row.contains("2026"),
            "row[{i}] contains leaked sync-mode param: {:?}",
            row
        );
        assert!(
            !row.contains("[2J"),
            "row[{i}] contains leaked erase-display: {:?}",
            row
        );
    }
}

#[test]
fn btop_truecolor_sgr_parsed() {
    let mut vt = avt::Vt::builder().size(80, 5).build();
    vt.feed_str("\x1b[38;2;255;100;0m\x1b[48;2;20;20;40mORANGE\x1b[0m");

    let line = vt.view().next().unwrap();
    let cells = line.cells();

    // The text cells should have the right chars
    let text: String = cells.iter()
        .take(6)
        .filter(|c| c.width() > 0)
        .map(|c| c.char())
        .collect();
    assert_eq!(text, "ORANGE", "truecolor SGR garbled text: {:?}", text);

    // The first cell should have truecolor foreground
    let pen = cells[0].pen();
    assert!(
        pen.foreground().is_some(),
        "expected fg color, got none"
    );
}
