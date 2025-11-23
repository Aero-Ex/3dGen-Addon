import blenderproc as bproc

# (rest of your pipeline script...)
bproc.init()
# install_addon.py
import bpy
import os

zip_path = os.path.expanduser('./extern/uvpackmaster/uvpackmaster3-addon-3.4.4-u3.zip')
bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
module_name = 'uvpackmaster3'


# Install (overwrite if already exists)
bpy.ops.preferences.addon_install(filepath=zip_path, overwrite=True)
bpy.ops.preferences.addon_enable(module=module_name)

