# GUI Redesign Design

## Summary

Redesign the PySide6 desktop GUI for the synthetic holography preview app to feel like a modern instrument-control workstation while preserving the existing processing workflow and data behavior.

The approved direction is:

- Layout: `Command Deck`
- Visual style: `Probe Studio`
- Control density: `compact`

This is a GUI-focused redesign centered in `viewer.py`. The processing pipeline and exported outputs remain unchanged.

## Goals

- Make the viewer feel more intentional, premium, and easier to scan during repeated lab use.
- Increase visual hierarchy so the plot area is clearly the primary focus.
- Group controls into a denser, more coherent control rail that matches the user workflow.
- Present diagnostics and manual tuning as instrument-like readouts rather than generic form fields.
- Preserve existing behaviors for loading, switching passage, changing harmonic/stage, tuning overrides, plotting, and exporting.

## Non-Goals

- No changes to hologram reconstruction or numerical processing logic.
- No changes to export formats, file naming, or processing semantics.
- No feature expansion beyond the current workflow.
- No migration away from PySide6 or Matplotlib.

## Current Context

The app currently provides:

- folder selection, loading, and export controls
- passage selection
- harmonic selection
- stage selection
- advanced processing controls and diagnostics
- an embedded Matplotlib viewer
- a log panel

The current implementation already has a custom dark theme, but the hierarchy is still relatively flat and the layout reads more like a generic utility window than a focused scientific instrument interface.

## Recommended Approach

Use a moderate layout refactor that reorganizes the existing UI into a clearer application shell while reusing the current widgets, event wiring, and processing flow.

This approach is preferred over:

- a visual-only reskin, which would not be enough to achieve the chosen instrument-control direction
- a fully bespoke control surface, which would introduce unnecessary complexity and implementation risk for this project

## Layout Design

### Primary Shell

The main window should be reorganized into two primary zones:

1. A left-aligned compact command rail
2. A right-side visualization workspace

The command rail should have a fixed width and contain dense, vertically stacked sections. The visualization workspace should expand to take the majority of the window and visually behave like the instrument display.

### Command Rail

The left rail should be split into three stacked sections:

### Dataset

Contains:

- folder path input
- browse action
- load action
- export action
- lightweight dataset or cache status text

This section should feel like session setup rather than a generic file picker row.

### Acquisition

Contains:

- passage selection
- harmonic selection
- stage selection
- processing mode selection

These controls should be visually grouped together because they define what is currently being inspected.

### Tuning

Contains:

- detected diagnostics values
- current override inputs for shift, width, and padding factor
- apply and reset actions

The current “show advanced” behavior should be replaced with a compact collapsible tuning section that is expanded by default after a dataset is loaded. This keeps tuning immediately available during active inspection while still allowing the rail to stay dense when the user wants to collapse it. The section should feel like precision adjustment controls, not a hidden secondary form.

### Visualization Workspace

The right side should contain:

- the primary amplitude/phase viewer in the most visually prominent panel
- the Matplotlib toolbar integrated visually with the viewer panel
- a bottom telemetry strip for status, metrics, and logs

The viewer panel should dominate the window so the data remains the clear focal point.

### Telemetry Strip

The bottom area below the viewer should consolidate short-form operational feedback:

- current status message
- compact diagnostics summary or key measurements
- log output panel

This should replace the current heavier log block with a shorter, more integrated information strip that still preserves scrolling log visibility. The log area should remain readable but visually secondary to the plots.

## Visual System

### Overall Tone

The chosen tone is `Probe Studio`: a warmer, premium scientific desktop aesthetic.

Characteristics:

- warm graphite panel surfaces
- restrained amber highlights
- compact, high-density information presentation
- softened corners instead of harsh industrial edges
- high contrast around the plot area so the visualization remains central

The interface should feel serious and technical, not playful or game-like.

### Color Direction

Use a restrained palette built around:

- dark graphite or charcoal backgrounds
- slightly lighter panel surfaces for sectional separation
- amber or gold accent highlights for active states and primary actions
- soft neutral text for labels
- brighter text or chips for active values and key readouts

Accent color usage should stay limited so that the viewer and active controls stand out clearly.

### Typography

Typography should support quick scanning in dense panels:

- section headers should feel more deliberate and instrument-like
- control labels should be concise and consistent
- numeric values should be easy to distinguish from labels
- the visual hierarchy should make active context obvious at a glance

The app does not need decorative typography, but it should move away from feeling default or purely utilitarian.

### Panel and Control Styling

Panels should read as grouped instrument modules:

- clearer section framing
- tighter internal spacing
- stronger distinction between labels, values, and action controls
- buttons that feel intentional and stateful
- input fields that look calibrated and precise

Primary actions such as `Load` and `Export All` should be visually differentiated without overpowering the plots.

## Interaction and Behavior

All existing interaction behavior should remain functionally consistent:

- loading a folder still validates and processes the selected dataset
- passage switching still reloads or reuses cached loaded state as it does today
- harmonic and stage changes still refresh the viewer
- processing mode changes still trigger the existing mode-switch behavior
- manual shift, width, and padding factor overrides still validate integer inputs and reload accordingly
- export still writes the same output set

The redesign should improve clarity, not alter the meaning of the controls.

### Padding Factor Control

The `Tuning` section should expose `Padding Factor` as an editable manual control alongside shift and width.

Requirements:

- `Padding Factor` accepts only positive integers
- the field displays the current loaded `pad_fact` value from processing settings
- `Apply` reloads using the current shift, width, and padding factor values together
- `Reset to Auto` clears manual overrides and restores `Padding Factor` to the default processing value of `1`
- validation errors for padding factor should use the same dialog and log pattern as the other tuning inputs

This should remain part of the existing tuning workflow rather than becoming a separate processing panel.

## Error Handling

Existing error-handling patterns should be retained:

- invalid folder selection remains a blocking error
- invalid manual override inputs remain blocking errors
- processing failures remain surfaced clearly

In addition, the status area should make failures and current state more visible without requiring users to infer them only from dialogs or a plain log block.

## Component Boundaries

The redesign should remain localized mainly to `viewer.py`, with the layout split into clearer UI-building units if needed.

Suggested component groupings inside the file:

- command rail builder
- dataset section builder
- acquisition section builder
- tuning section builder
- viewer workspace builder
- telemetry strip builder
- theme helpers

This keeps the redesign readable and reduces the risk of tangling layout changes with processing behavior.

## Data Flow

No data-flow changes are required.

The GUI should continue to use:

- `load_passage` for dataset loading
- `get_view_image` for rendering the selected stage and harmonic
- `export_all_views` for export
- the existing cached loaded-state and override structures, extended to include `pad_fact`

Only the presentation and grouping of controls should change.

## Testing and Validation

Validation should include:

- running the existing automated test suite
- launching the desktop app and checking that the window renders correctly
- verifying load, passage switch, harmonic switch, stage switch, manual apply/reset, and export
- verifying that `Padding Factor` shows the loaded `pad_fact` value
- verifying that invalid or non-positive `Padding Factor` values are rejected
- verifying that `Apply` passes `Padding Factor` into reloads
- verifying that `Reset to Auto` restores `Padding Factor` to `1`
- visually checking both desktop readability and smaller-window behavior
- confirming that Matplotlib integration still works after layout changes

## Risks

Primary risks:

- breaking signal connections while moving widgets into a new layout
- reducing readability if the compact control rail becomes too compressed
- making the log or diagnostics too hidden in pursuit of visual polish
- introducing visual inconsistency between Qt widgets and the embedded Matplotlib toolbar/canvas

Mitigations:

- preserve existing widget behavior and signal wiring wherever possible
- refactor layout structure incrementally rather than rewriting logic
- keep diagnostics visible and scannable even in compact form
- explicitly restyle the Matplotlib toolbar to belong to the new viewer surface

## Implementation Notes for the Next Phase

The implementation plan should focus on:

1. extracting or reorganizing UI sections in `viewer.py`
2. replacing the current top bars and sidebar composition with the approved shell
3. updating theme tokens and widget styling to the `Probe Studio` direction
4. refining the viewer and telemetry areas
5. validating behavior with tests and manual launch checks

## Success Criteria

The redesign is successful when:

- the app clearly reads as a modern scientific instrument interface
- the viewer is the obvious focal point
- the left-side control flow is dense but still easy to scan
- diagnostics and tuning feel integrated instead of bolted on
- all existing core behaviors remain intact
