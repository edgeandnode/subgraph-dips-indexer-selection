from datetime import datetime

import pytest
from freezegun import freeze_time

from iisa.time import derive_timestamps


class TestDeriveTimestamps:
    """
    Tests for the derive_timestamps function.

    This class tests various scenarios for the derive_timestamps function,
    including positive days, zero days, negative days, and non-integer inputs.
    It also verifies the correctness of the returned types and formats.
    """

    @freeze_time("2024-08-05 12:00:00")
    def test_with_positive_days(self):
        """
        Test derive_timestamps with a positive number of days.
        """
        start_date, end_date, start_ts, end_ts = derive_timestamps(7)

        assert end_date == datetime(2024, 8, 5, 12, 0, 0)
        assert start_date == datetime(2024, 7, 29, 12, 0, 0)
        assert end_ts == "2024-08-05T12:00:00Z"
        assert start_ts == "2024-07-29T12:00:00Z"

    def test_with_zero_days(self):
        """
        Test derive_timestamps with zero days.
        """
        start_date, end_date, start_ts, end_ts = derive_timestamps(0)

        # Start and end dates should be the same.
        assert start_date == end_date
        assert start_ts == end_ts

    def test_with_negative_days(self):
        """
        Test derive_timestamps with negative days.
        """
        # Should raise a ValueError when given a negative number of days.
        with pytest.raises(ValueError, match="num_days must be a non-negative integer"):
            derive_timestamps(-1)

    def test_derive_timestamp_format(self):
        """
        Test the format of timestamps returned by derive_timestamps.
        """
        # Timestamp format should be consistent and reversible
        _, _, start_ts, end_ts = derive_timestamps(1)
        date_format = "%Y-%m-%dT%H:%M:%SZ"
        datetime.strptime(start_ts, date_format)
        datetime.strptime(end_ts, date_format)
