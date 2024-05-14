"""
Please install fuxictr first, or directly add the package to sys.path
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([current_dir])
import fuxictr
assert fuxictr.__version__ == "2.2.3"
