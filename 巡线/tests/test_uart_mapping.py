import importlib
import struct
import sys
import types
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

maix = types.ModuleType("maix")
for name in ("app", "err", "pinmap", "thread", "time", "uart"):
    setattr(maix, name, types.SimpleNamespace())
sys.modules.setdefault("maix", maix)

uart_link = importlib.import_module("uart_link")


class UartNearFarMappingTests(unittest.TestCase):
    def test_publish_snapshot_maps_near_to_x_and_far_to_y(self):
        link = uart_link.LineFollowLink()
        link._started = True

        link.publish_line_data(
            near_error=12.7,
            far_error=-34.2,
            confidence=0.8,
            fps=31.5,
        )

        self.assertEqual(
            link._target_snapshot,
            (1, 12, -34, uart_link.FLAG_TARGET_VALID, 31.5),
        )

    def test_frame_remains_32_bytes_and_contains_both_errors(self):
        frame = uart_link.build_frame(
            packet_sequence=7,
            vision_frame=9,
            timestamp_ms=1234,
            x_error=25,
            y_error=-40,
            flags=uart_link.FLAG_TARGET_VALID,
        )

        self.assertEqual(len(frame), 32)
        values = struct.unpack(uart_link._FRAME_FORMAT, frame[:30])
        self.assertEqual(values[7], 25)
        self.assertEqual(values[8], -40)
        self.assertEqual(
            struct.unpack("<H", frame[30:])[0],
            uart_link.crc16_ccitt_false(frame[:30]),
        )


if __name__ == "__main__":
    unittest.main()
