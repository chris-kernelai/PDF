"""
image_description_integrator.py

Wrapper module that imports from 5_integrate_descriptions.py
This provides a clean import name for the ImageDescriptionIntegrator class.
"""

# Import all public symbols from the main module
import sys
import importlib.util

# Load the module from file
spec = importlib.util.spec_from_file_location(
    "integrate_descriptions_module",
    "5_integrate_descriptions.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules["integrate_descriptions_module"] = module
spec.loader.exec_module(module)

# Export the main class
ImageDescriptionIntegrator = module.ImageDescriptionIntegrator

__all__ = ["ImageDescriptionIntegrator"]
