def _fmt(label, val):
    """Format a KPI value based on its label."""
    if label == "Revenue": return f"€{val:,.0f}"
    if label == "CR":      return f"{val:.2%}"
    if label == "AOV":     return f"€{val:.2f}"
    return f"{val:,.0f}"


def color_neg(val):
    """Return red CSS if negative, green if non-negative."""
    if isinstance(val, (int, float)) and val < 0:
        return "color: red; font-weight: bold"
    return "color: green"
