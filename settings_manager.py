import bpy
import json
import os

def get_settings_path():
    """Get the path to the settings file"""
    addon_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(addon_dir, "settings.json")

def save_settings(context):
    """Save current settings to JSON file"""
    try:
        props = context.scene.trellis_props
        settings = {
            "generation_mode": props.generation_mode,
            "sparse_steps": props.sparse_steps,
            "sparse_cfg": props.sparse_cfg,
            "slat_steps": props.slat_steps,
            "slat_cfg": props.slat_cfg,
            "seed": props.seed,
            "preprocess_image": props.preprocess_image,
            "texture_size": int(props.texture_size),
            "simplify_mesh": props.simplify_mesh,
            "generate_texture": props.generate_texture,
        }
        
        with open(get_settings_path(), 'w') as f:
            json.dump(settings, f, indent=4)
            
        print(f"TRELLIS: Settings saved to {get_settings_path()}")
    except Exception as e:
        print(f"TRELLIS: Failed to save settings: {e}")
        import traceback
        traceback.print_exc()

def load_settings(context):
    """Load settings from JSON file"""
    try:
        settings_path = get_settings_path()
        if not os.path.exists(settings_path):
            print("TRELLIS: No saved settings found")
            return
            
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            
        props = context.scene.trellis_props
        
        # Load properties safely
        if "generation_mode" in settings:
            props.generation_mode = settings["generation_mode"]
        if "sparse_steps" in settings:
            props.sparse_steps = settings["sparse_steps"]
        if "sparse_cfg" in settings:
            props.sparse_cfg = settings["sparse_cfg"]
        if "slat_steps" in settings:
            props.slat_steps = settings["slat_steps"]
        if "slat_cfg" in settings:
            props.slat_cfg = settings["slat_cfg"]
        if "seed" in settings:
            props.seed = settings["seed"]
        if "preprocess_image" in settings:
            props.preprocess_image = settings["preprocess_image"]
        if "texture_size" in settings:
            props.texture_size = str(settings["texture_size"])
        if "simplify_mesh" in settings:
            props.simplify_mesh = settings["simplify_mesh"]
        if "generate_texture" in settings:
            props.generate_texture = settings["generate_texture"]
            
        print(f"TRELLIS: Settings loaded from {settings_path}")
    except Exception as e:
        print(f"TRELLIS: Failed to load settings: {e}")
        import traceback
        traceback.print_exc()
