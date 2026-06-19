# Built-in label maps + recommended reconstruction mode per task. A preset maps
# integer label values to anatomical names used for the output STL filenames,
# and declares whether the structure should be reconstructed by CSG peel (one
# hull partitioned by internal interfaces) or independently (disjoint organs).

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    """A task preset: label->name map plus the recommended reconstruction mode."""
    mode: str                # "csg" or "independent"
    labels: dict             # {int label: str name}
    description: str = ""


PRESETS = {
    # One structure partitioned by internal fissures/interfaces -> CSG peel.
    "lung_lobe": Preset(
        mode="csg",
        labels={
            1: "left_lower_lobe", 2: "left_upper_lobe",
            3: "right_lower_lobe", 4: "right_middle_lobe", 5: "right_upper_lobe",
        },
        description="5 pulmonary lobes (shared fissure seams).",
    ),
    "lung_segment": Preset(
        mode="csg",
        labels={
            1: "LS1_2", 2: "LS3", 3: "LS4", 4: "LS5",
            5: "LS6", 6: "LS8", 7: "LS9", 8: "LS10",
            9: "RS1", 10: "RS2", 11: "RS3", 12: "RS4",
            13: "RS5", 14: "RS6", 15: "RS7", 16: "RS8",
            17: "RS9", 18: "RS10",
        },
        description="18 bronchopulmonary segments.",
    ),
    "couinaud": Preset(
        mode="csg",
        labels={
            1: "S1", 2: "S2", 3: "S3", 4: "S4",
            5: "S5", 6: "S6", 7: "S7", 8: "S8",
        },
        description="8 Couinaud liver segments.",
    ),
    # Disjoint organs -> independent surfaces (no shared cut plane).
    "core_organs": Preset(
        mode="independent",
        labels={
            1: "liver", 2: "pancreas", 3: "spleen",
            4: "left_kidney", 5: "right_kidney",
        },
        description="Disjoint abdominal organs.",
    ),
    "vessel": Preset(
        mode="independent",
        labels={1: "PV", 2: "HV", 3: "HA"},
        description="Portal vein / hepatic vein / hepatic artery.",
    ),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Available: {', '.join(sorted(PRESETS))}")
    return PRESETS[name]
