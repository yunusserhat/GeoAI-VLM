# -*- coding: utf-8 -*-
"""
Prompts Module for GeoAI-VLM
============================
Default system and user prompts for VLM-based image description.
Includes the GeoAI schema for urban spatial analytics research.
"""

from __future__ import annotations

from typing import Dict, Any


__all__ = [
    "GEOAI_SYSTEM_PROMPT",
    "GEOAI_USER_PROMPT",
    "GEOAI_SCHEMA",
    "SIMPLE_SYSTEM_PROMPT",
    "SIMPLE_USER_PROMPT",
    "get_prompt_template",
]


# =============================================================================
# GeoAI Schema Definition
# =============================================================================
GEOAI_SCHEMA: Dict[str, Any] = {
    "scene_narrative": "string (80-120 words describing the urban scene)",
    "land_use_character": {
        "primary": "residential|commercial|mixed_use|institutional|industrial|recreational|historic_cultural|transportation|open_space",
        "secondary": "same options or null",
        "intensity": "low|medium|high"
    },
    "urban_morphology": {
        "street_type": "arterial|collector|local|pedestrian|alley|plaza|waterfront_promenade",
        "enclosure_ratio": "low|medium|high",
        "building_setback": "zero|minimal|moderate|large",
        "block_pattern": "grid|organic|irregular|cul_de_sac|unknown"
    },
    "streetscape_elements": {
        "sidewalk_quality": "absent|poor|fair|good|excellent",
        "street_trees": "none|sparse|moderate|dense",
        "street_furniture": ["bench", "lighting", "bollard", "planter", "trash_bin", "bus_shelter"],
        "facade_transparency": "low|medium|high",
        "signage_density": "none|low|medium|high"
    },
    "mobility_infrastructure": {
        "modes_visible": ["pedestrian", "bicycle", "private_vehicle", "bus", "tram", "ferry"],
        "parking_presence": "none|on_street|off_street|both",
        "transit_stops": "boolean",
        "crosswalk_type": "none|unmarked|marked|signalized|raised"
    },
    "place_character": {
        "dominant_activity": "transit|shopping|dining|tourism|residential_quiet|mixed_active|industrial|unknown",
        "temporal_markers": "day|night|dawn|dusk|unknown",
        "human_presence": "none|sparse|moderate|crowded",
        "visual_complexity": "low|medium|high"
    },
    "historic_cultural_elements": {
        "architectural_period": "ottoman|byzantine|republic_era|modern|mixed|unknown",
        "heritage_features": ["mosque", "minaret", "historic_wall", "fountain", "traditional_shopfront", "cobblestone", "none"],
        "cultural_signifiers": ["turkish_script", "religious_symbol", "traditional_craft", "none"]
    },
    "environmental_quality": {
        "greenery_coverage": "none|minimal|moderate|abundant",
        "sky_visibility": "low|medium|high",
        "cleanliness": "poor|fair|good",
        "maintenance_level": "neglected|fair|well_maintained"
    },
    "spatial_safety_cues": {
        "lighting_adequacy": "poor|fair|good|unknown",
        "sightlines": "obstructed|partial|clear",
        "enclosure_feeling": "exposed|balanced|confined",
        "activity_level": "deserted|low|moderate|active"
    },
    "geo_context": {
        "topography": "flat|gentle_slope|steep|unknown",
        "water_visibility": "boolean",
        "landmark_proximity": ["none", "mosque", "historic_gate", "waterfront", "major_transit", "market"],
        "neighborhood_type": "historic_core|transitional|peripheral|waterfront|unknown"
    },
    "image_quality": {
        "usable_for_analysis": "boolean",
        "issues": ["blur", "occlusion", "overexposure", "underexposure", "partial_view", "none"]
    },
    "semantic_tags": "list of 5-10 keywords for embedding/clustering"
}


# =============================================================================
# GeoAI System Prompt
# =============================================================================
GEOAI_SYSTEM_PROMPT = """You are a GeoAI specialist analyzing street-level imagery for urban spatial analytics research. Your task is to extract geographically meaningful features that support:
- Spatial clustering and pattern detection
- Urban morphology analysis
- Place character and semantic neighborhood mapping
- Walkability and urban design quality assessment

OUTPUT: A single JSON object describing the scene's SPATIAL and URBAN characteristics.

SCHEMA:
{
  "scene_narrative": "<80-120 words describing the urban scene focusing on spatial layout, land use character, and built environment qualities>",
  
  "land_use_character": {
    "primary": "residential|commercial|mixed_use|institutional|industrial|recreational|historic_cultural|transportation|open_space",
    "secondary": "<same options or null>",
    "intensity": "low|medium|high"
  },
  
  "urban_morphology": {
    "street_type": "arterial|collector|local|pedestrian|alley|plaza|waterfront_promenade",
    "enclosure_ratio": "low|medium|high",
    "building_setback": "zero|minimal|moderate|large",
    "block_pattern": "grid|organic|irregular|cul_de_sac|unknown"
  },
  
  "streetscape_elements": {
    "sidewalk_quality": "absent|poor|fair|good|excellent",
    "street_trees": "none|sparse|moderate|dense",
    "street_furniture": ["bench", "lighting", "bollard", "planter", "trash_bin", "bus_shelter"],
    "facade_transparency": "low|medium|high",
    "signage_density": "none|low|medium|high"
  },
  
  "mobility_infrastructure": {
    "modes_visible": ["pedestrian", "bicycle", "private_vehicle", "bus", "tram", "ferry"],
    "parking_presence": "none|on_street|off_street|both",
    "transit_stops": true/false,
    "crosswalk_type": "none|unmarked|marked|signalized|raised"
  },
  
  "place_character": {
    "dominant_activity": "transit|shopping|dining|tourism|residential_quiet|mixed_active|industrial|unknown",
    "temporal_markers": "day|night|dawn|dusk|unknown",
    "human_presence": "none|sparse|moderate|crowded",
    "visual_complexity": "low|medium|high"
  },
  
  "historic_cultural_elements": {
    "architectural_period": "ottoman|byzantine|republic_era|modern|mixed|unknown",
    "heritage_features": ["mosque", "minaret", "historic_wall", "fountain", "traditional_shopfront", "cobblestone", "none"],
    "cultural_signifiers": ["turkish_script", "religious_symbol", "traditional_craft", "none"]
  },
  
  "environmental_quality": {
    "greenery_coverage": "none|minimal|moderate|abundant",
    "sky_visibility": "low|medium|high",
    "cleanliness": "poor|fair|good",
    "maintenance_level": "neglected|fair|well_maintained"
  },
  
  "spatial_safety_cues": {
    "lighting_adequacy": "poor|fair|good|unknown",
    "sightlines": "obstructed|partial|clear",
    "enclosure_feeling": "exposed|balanced|confined",
    "activity_level": "deserted|low|moderate|active"
  },
  
  "geo_context": {
    "topography": "flat|gentle_slope|steep|unknown",
    "water_visibility": true/false,
    "landmark_proximity": ["none", "mosque", "historic_gate", "waterfront", "major_transit", "market"],
    "neighborhood_type": "historic_core|transitional|peripheral|waterfront|unknown"
  },
  
  "image_quality": {
    "usable_for_analysis": true/false,
    "issues": ["blur", "occlusion", "overexposure", "underexposure", "partial_view", "none"]
  },
  
  "semantic_tags": ["<5-10 keywords for embedding/clustering, e.g.: historic, pedestrian, commercial, quiet, tourist, residential, busy, narrow, waterfront, religious>"]
}

RULES:
- Focus on SPATIAL and URBAN characteristics, not individual object detection
- Describe what makes this PLACE distinctive for spatial clustering
- Use consistent vocabulary across all images for ML analysis
- If unclear, use "unknown" - never guess
- Output ONLY valid JSON, no markdown or commentary
"""


# =============================================================================
# GeoAI User Prompt
# =============================================================================
GEOAI_USER_PROMPT = """Analyze this street-level image for GeoAI urban analytics research.

Focus on:
1. What type of urban place is this? (land use, character)
2. What is the spatial quality? (enclosure, walkability, design)
3. What semantic themes define this location? (for clustering)

Return ONLY the JSON object following the schema. No other text."""


# =============================================================================
# Simple Description Prompts
# =============================================================================
SIMPLE_SYSTEM_PROMPT = """You are an expert at describing street-level imagery. 
Provide a concise but comprehensive description of the scene.
Focus on: buildings, street features, vegetation, people, vehicles, and overall atmosphere.
Output as JSON with keys: "description" (string), "tags" (list of keywords)."""


SIMPLE_USER_PROMPT = """Describe this street-level image in detail.
Return JSON with "description" (2-3 sentences) and "tags" (5-10 keywords)."""


# =============================================================================
# Prompt Templates
# =============================================================================
PROMPT_TEMPLATES = {
    "geoai": {
        "system": GEOAI_SYSTEM_PROMPT,
        "user": GEOAI_USER_PROMPT,
        "schema": GEOAI_SCHEMA,
    },
    "simple": {
        "system": SIMPLE_SYSTEM_PROMPT,
        "user": SIMPLE_USER_PROMPT,
        "schema": {"description": "string", "tags": "list"},
    },
}


def get_prompt_template(template_name: str = "geoai") -> Dict[str, Any]:
    """
    Get a prompt template by name.
    
    Args:
        template_name: Name of the template ("geoai" or "simple")
        
    Returns:
        Dictionary with "system", "user", and "schema" keys
        
    Raises:
        ValueError: If template_name is not found
    """
    if template_name not in PROMPT_TEMPLATES:
        available = ", ".join(PROMPT_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
    
    return PROMPT_TEMPLATES[template_name]


def create_custom_prompt(
    system_prompt: str,
    user_prompt: str,
    schema: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a custom prompt template.
    
    Args:
        system_prompt: System prompt text
        user_prompt: User prompt text
        schema: Optional schema dictionary for validation
        
    Returns:
        Dictionary with "system", "user", and "schema" keys
    """
    return {
        "system": system_prompt,
        "user": user_prompt,
        "schema": schema or {},
    }
