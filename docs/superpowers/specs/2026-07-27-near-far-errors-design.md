# Near/Far Error Design

## Goal

Extend the MaixCAM2 line-following output with independent near-field and far-field lateral errors while preserving the existing 32-byte V2 UART frame.

## Region definition

- Keep `NUM_STRIPS = 15`.
- Use the top five configured strips as the far region.
- Use the bottom five configured strips as the near region.
- Select regions by configured strip Y coordinates, not by the number of currently detected points, so the physical image regions do not move when detections are missing.

## Error calculation

- Compute each region's line center using the existing per-strip weights.
- Define positive error as line center to the right of the image center and negative error as line center to the left.
- Apply the existing `SMOOTH_FACTOR` independently to `near_error` and `far_error`.
- When a region has no valid point in the current frame, keep that region's last filtered error.
- Keep the existing overall `error` and overall `confidence` for compatibility and display.

## UART mapping

- Preserve frame size, version, layout, and CRC.
- Map `near_error` to the existing `x_error` int16 field.
- Map `far_error` to the existing `y_error` int16 field.
- Keep the target-valid flag based on the existing overall confidence threshold.
- Update debug logging to label the two fields explicitly.

## Testing

Add a hardware-independent helper module and unit tests covering region selection, weighted averaging, sign convention, sparse detections, and missing-region behavior. Run the tests with CPython; Maix hardware APIs are not required for these calculations.
