"""Reference knowledge for the seven damage classes.

This is general conservation-literature background (e.g. ICOMOS-ISCS
"Illustrated glossary on stone deterioration patterns", 2008). It is shown as
*reference*, never as a diagnosis of a specific monument. Monument-specific
hypotheses come from Gemini (core/explain.py) and are labelled as AI-generated.
"""

DAMAGE = {
    "Crack": {
        "color": "#C0392B", "marker": "diamond", "group": "Structural",
        "what": "A visible fissure separating parts of the masonry or render; may be hairline or open.",
        "typical_causes": ["Structural stress or differential settlement", "Thermal expansion and contraction cycles",
                           "Water infiltration and freeze/salt crystallisation in pores", "Vibration (traffic, construction)",
                           "Ageing and loss of binder in mortar"],
        "prevention": ["Map and date cracks; monitor width with tell-tales or crack gauges",
                       "Control water ingress at roofs, joints and drainage",
                       "Refer active or widening cracks to a structural/heritage engineer"],
    },
    "Discolouration": {
        "color": "#B9770E", "marker": "circle", "group": "Surface",
        "what": "Change of the stone's original colour (darkening, staining, yellowing).",
        "typical_causes": ["Deposition of airborne pollutants (soot, particulates)", "Iron or copper staining from fixtures",
                           "Biological films", "Previous inappropriate treatments or coatings"],
        "prevention": ["Reduce local emission sources around the site", "Gentle, tested cleaning by conservators",
                       "Remove corroding metal fixtures"],
    },
    "Efflorescence": {
        "color": "#7D3C98", "marker": "square", "group": "Moisture",
        "what": "Whitish, powdery salt deposits on the surface left as moisture evaporates.",
        "typical_causes": ["Rising damp from the ground", "Rain penetration and poor drainage",
                           "Soluble salts in mortar, soil or cement repairs", "Wet–dry cycling"],
        "prevention": ["Identify and cut off the moisture source", "Improve site drainage",
                       "Avoid cement-based repairs; use compatible lime mortars", "Poultice desalination by specialists"],
    },
    "Fabric Loss of Masonry Units": {
        "color": "#1F618D", "marker": "x", "group": "Structural",
        "what": "Missing material — detached, fallen or eroded stones, bricks or ornament.",
        "typical_causes": ["Advanced weathering and erosion", "Mechanical impact or vandalism",
                           "Failure of mortar joints", "Root action of vegetation", "Long-term structural movement"],
        "prevention": ["Secure loose elements and restrict access below", "Repoint joints with compatible mortar",
                       "Document losses photographically for authentic reconstruction decisions"],
    },
    "Moss": {
        "color": "#1E8449", "marker": "circle-open", "group": "Biological",
        "what": "Moss or bryophyte colonisation, usually on shaded, damp surfaces.",
        "typical_causes": ["Persistent surface moisture", "Shade and poor ventilation",
                           "Water run-off from cornices and gutters", "Accumulated soil/dust in joints"],
        "prevention": ["Fix the water run-off path", "Careful manual removal by conservators",
                       "Only conservator-approved biocides"],
    },
    "Plant": {
        "color": "#117864", "marker": "triangle-up", "group": "Biological",
        "what": "Higher plants (grass, shrubs, saplings) rooted in the masonry.",
        "typical_causes": ["Seeds carried by birds/wind into open joints", "Moisture retention in cracks and joints",
                           "Lack of routine maintenance"],
        "prevention": ["Remove saplings early, cutting rather than pulling roots", "Repoint open joints",
                       "Schedule periodic vegetation clearance"],
    },
    "Surface Dissolution": {
        "color": "#5D6D7E", "marker": "cross", "group": "Surface",
        "what": "Loss of surface by chemical dissolution — rounding of edges, sugaring, pitting.",
        "typical_causes": ["Acidic rain and pollutant gases (SO₂, NOₓ) reacting with calcareous stone",
                           "Long-term exposure to run-off water", "Salt weathering"],
        "prevention": ["Air-quality management around the monument", "Protective shelter/drip details for run-off",
                       "Conservator-tested consolidants only where justified"],
    },
}


def color(label: str) -> str:
    return DAMAGE.get(label, {}).get("color", "#0B1F3A")
