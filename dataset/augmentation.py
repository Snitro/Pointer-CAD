import random
import re

# Unit config: abbreviation, full name, conversion scale, sampling probability.
UNIT_CONFIG = {
    "m": {"full": "meter", "scale": 1, "prob": 0.1},
    "cm": {"full": "centimeter", "scale": 100, "prob": 0.3},
    "mm": {"full": "millimeter", "scale": 1000, "prob": 0.6},
}


def enhance_text_with_units_en(
    text,
    target_unit="millimeter",
    scale_factor=1000,
    add_space=False,
    value_multiplier=1,
):
    def format_float(val):
        return f"{val:.5f}".rstrip("0").rstrip(".") if "." in f"{val:.5f}" else str(val)

    def replace_fn(match):
        original_val = float(match.group(1))
        new_val = original_val * scale_factor * value_multiplier
        formatted_val = format_float(new_val)
        space = " " if add_space else ""
        return f"{formatted_val}{space}{target_unit}"

    return re.sub(r"<v>(-?[\d\.]+)</v>", replace_fn, text)


def randomly_enhance_cad_data(prompt, scale, movement):
    """
    Randomly augment CAD text and parameters.

    Args:
        prompt: Input text containing <v>number</v> tags.
        scale (float): Scale factor.
        movement (list of float): Translation parameters (3 values).

    Returns:
        Tuple: (enhanced_prompt, scaled_scale, scaled_translation)
    """
    units = list(UNIT_CONFIG.keys())
    probs = [UNIT_CONFIG[u]["prob"] for u in units]

    selected_unit_abbr = random.choices(units, weights=probs, k=1)[0]
    unit_info = UNIT_CONFIG[selected_unit_abbr]

    target_unit = random.choice([selected_unit_abbr, unit_info["full"]])
    scale_factor = unit_info["scale"]

    value_multiplier = random.uniform(0.5, 2.0)
    add_space = random.choice([True, False])

    enhanced_prompt = enhance_text_with_units_en(
        text=prompt,
        target_unit=target_unit,
        scale_factor=scale_factor,
        add_space=add_space,
        value_multiplier=value_multiplier,
    )

    scaled_scale = float(scale * value_multiplier)
    scaled_translation = [float(t * value_multiplier) for t in movement]

    return enhanced_prompt, scaled_scale, scaled_translation


def format_cad_data(prompt, scale, movement):
    """
    Deterministically format CAD text and parameters.

    Args:
        prompt: Raw text with <v>number</v> tags.
        scale (float): Original scale factor.
        movement (list of float): Translation vector.

    Returns:
        Tuple: (enhanced_prompt, scaled_scale, scaled_translation)
    """

    value_multiplier = 1
    add_space = False

    # Auto-select unit from m and mm so scale * scale_factor >= 1.
    for unit_key in ["m", "mm"]:
        scale_factor = UNIT_CONFIG[unit_key]["scale"]
        if scale * scale_factor * value_multiplier >= 1:
            break

    target_unit = unit_key

    enhanced_prompt = enhance_text_with_units_en(
        text=prompt,
        target_unit=target_unit,
        scale_factor=scale_factor,
        add_space=add_space,
        value_multiplier=value_multiplier,
    )

    return enhanced_prompt, scale, movement
