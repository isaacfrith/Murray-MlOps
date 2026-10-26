"""Turn a forecast day (level, level change, wind) into good / marginal / no-go (pure)."""

from dataclasses import dataclass, field
from importlib.resources import files

import yaml

BANDS = ["good", "marginal", "no-go"]  # ordered best → worst
NO_FORECAST = "no forecast"


@dataclass
class Rating:
    rating: str
    reasons: list[str] = field(default_factory=list)


def load_rule() -> dict:
    return yaml.safe_load(files("murray.rules").joinpath("wakeboarding.yaml").read_text())


def _worse(a: str, b: str) -> str:
    return max(a, b, key=BANDS.index)


def rate_day(level_m: float | None, change_m: float | None, wind_kmh: float | None,
             rule: dict) -> Rating:
    lv, wd = rule["level_m"], rule["wind_max_kmh"]
    if level_m is None or level_m != level_m:  # None or NaN
        return Rating(NO_FORECAST, ["no level forecast"])
    if level_m > lv["no_forecast_above"]:
        return Rating(NO_FORECAST, [f"flood conditions (level > {lv['no_forecast_above']:.2f} m)"])

    reasons = []
    if level_m < lv["no_go_below"]:
        band = "no-go"; reasons.append(f"low water {level_m:.2f} m (snags)")
    elif level_m > lv["no_go_above"]:
        band = "no-go"; reasons.append(f"high water {level_m:.2f} m")
    elif lv["good_min"] <= level_m <= lv["good_max"]:
        band = "good"; reasons.append(f"level {level_m:.2f} m in good range")
    else:
        band = "marginal"; reasons.append(f"level {level_m:.2f} m outside tried range")

    if wind_kmh is None or wind_kmh != wind_kmh:
        band = _worse(band, "marginal"); reasons.append("no wind forecast")
    elif wind_kmh > wd["marginal_max"]:
        band = "no-go"; reasons.append(f"strong wind {wind_kmh:.0f} km/h")
    elif wind_kmh > wd["good_max"]:
        band = _worse(band, "marginal"); reasons.append(f"breezy {wind_kmh:.0f} km/h")
    else:
        reasons.append(f"wind {wind_kmh:.0f} km/h")

    if change_m is not None and abs(change_m) > rule["fast_change_m_per_day"]:
        band = BANDS[min(BANDS.index(band) + 1, len(BANDS) - 1)]
        reasons.append(f"level changing fast ({change_m * 100:+.0f} cm/day)")
    return Rating(band, reasons)
