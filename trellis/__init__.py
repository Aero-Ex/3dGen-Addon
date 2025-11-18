# Handle imports for both package and standalone contexts
try:
    # Relative imports (normal package usage)
    from . import models
    from . import modules
    from . import pipelines
    from . import renderers
    from . import representations
    from . import utils
except (ImportError, ValueError) as e:
    # If relative imports fail, try absolute imports
    # This can happen in some console execution contexts
    import sys
    import os
    # Add current directory to path if not already there
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    # Try again with absolute imports
    try:
        import trellis.models as models
        import trellis.modules as modules
        import trellis.pipelines as pipelines
        import trellis.renderers as renderers
        import trellis.representations as representations
        import trellis.utils as utils
    except ImportError as import_error:
        # Re-raise with more context
        raise ImportError(f"Failed to import trellis modules: {import_error}. Original error: {e}")
