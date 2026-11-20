"""Rule-based AQI health-tier mapping (PRD TR-007 / BR-003).

Standard: CPCB National Air Quality Index (India). Tiers are assigned with plain
if/elif thresholds on the *predicted* AQI, exactly as the PRD's LLD specifies.
"""
from dataclasses import dataclass

# Upper bound of each tier (inclusive). Used for borderline detection too.
TIER_BOUNDARIES = [50, 100, 200, 300, 400]


@dataclass(frozen=True)
class HealthTier:
    label: str
    range_text: str
    color: str
    text_color: str
    icon: str
    severity: int
    health_impact: str          # CPCB's published health-impact statement
    general_public: str
    sensitive_groups: str


GOOD = HealthTier("Good", "0–50", "#2E9E5B", "#FFFFFF", "●", 1,
                  "Minimal impact.",
                  "Air quality is fine for normal outdoor activity.",
                  "No special precautions needed.")
SATISFACTORY = HealthTier("Satisfactory", "51–100", "#8DBF3F", "#16181D", "●", 2,
                          "Minor breathing discomfort to sensitive people.",
                          "Normal outdoor activity is fine.",
                          "People with respiratory conditions should watch for symptoms.")
MODERATE = HealthTier("Moderate", "101–200", "#E8B62C", "#16181D", "▲", 3,
                      "Breathing discomfort to people with lung disease such as asthma, and heart disease.",
                      "Consider reducing long or intense outdoor exertion.",
                      "People with asthma, lung or heart disease should limit prolonged outdoor exertion.")
POOR = HealthTier("Poor", "201–300", "#E8812C", "#16181D", "▲", 4,
                  "Breathing discomfort to most people on prolonged exposure.",
                  "Reduce prolonged or heavy outdoor exertion.",
                  "Children, older adults and people with heart or lung disease should avoid prolonged outdoor exertion.")
VERY_POOR = HealthTier("Very Poor", "301–400", "#D9443A", "#FFFFFF", "◆", 5,
                       "Respiratory illness on prolonged exposure.",
                       "Avoid prolonged outdoor exertion; consider moving activities indoors.",
                       "Sensitive groups should avoid outdoor physical activity.")
SEVERE = HealthTier("Severe", "401–500", "#8E2C48", "#FFFFFF", "◆", 6,
                    "Affects healthy people and seriously impacts those with existing diseases.",
                    "Avoid outdoor physical activity.",
                    "Sensitive groups should remain indoors and keep activity levels low.")

ALL_TIERS = [GOOD, SATISFACTORY, MODERATE, POOR, VERY_POOR, SEVERE]


def get_health_tier(predicted_aqi: float) -> HealthTier:
    """Map a numeric AQI to its CPCB health tier using if/elif threshold rules."""
    if predicted_aqi <= 50:
        return GOOD
    elif predicted_aqi <= 100:
        return SATISFACTORY
    elif predicted_aqi <= 200:
        return MODERATE
    elif predicted_aqi <= 300:
        return POOR
    elif predicted_aqi <= 400:
        return VERY_POOR
    else:
        return SEVERE


def nearest_boundary(aqi: float):
    """Return (boundary, distance) for the closest tier boundary."""
    b = min(TIER_BOUNDARIES, key=lambda x: abs(aqi - x))
    return b, abs(aqi - b)


def advisory_message(aqi: float, city: str | None = None) -> str:
    """Health advisory text generated from the predicted tier (PRD BR-008)."""
    t = get_health_tier(aqi)
    where = f" in {city}" if city else ""
    return (f"Predicted AQI{where} is {aqi:.0f}, which falls in the {t.label.upper()} tier ({t.range_text}). "
            f"Health impact (CPCB): {t.health_impact} General public: {t.general_public} "
            f"Sensitive groups: {t.sensitive_groups}")
