import math

from murray.rules.wakeboarding import load_rule, rate_day

RULE = load_rule()


def r(level, change=0.0, wind=15.0):
    return rate_day(level, change, wind, RULE).rating


def test_trip_days_rate_good():
    # The five logged trips (level m, 1-day change m, wind km/h): all were good.
    for level, change, wind in [(2.98, -0.074, 14.4), (2.35, 0.014, 20.7), (2.36, 0.011, 22.3),
                                (2.84, 0.017, 14.3), (2.84, -0.004, 14.4)]:
        assert r(level, change, wind) == "good", (level, change, wind)


def test_level_bands():
    assert r(1.9) == "no-go"
    assert r(2.1) == "marginal"
    assert r(3.2) == "marginal"
    assert r(3.6) == "no-go"
    assert r(4.2) == "no forecast"
    assert r(None) == "no forecast" and r(math.nan) == "no forecast"


def test_wind_and_fast_change():
    assert r(2.6, wind=25) == "marginal"
    assert r(2.6, wind=35) == "no-go"
    assert r(2.6, wind=None) == "marginal"
    assert r(2.6, change=0.15) == "marginal"       # good → marginal
    assert r(3.2, change=-0.15) == "no-go"         # marginal → no-go
