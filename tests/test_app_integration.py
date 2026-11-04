"""
Integration test for the full Streamlit application using Streamlit's AppTest.
"""

import os
from streamlit.testing.v1 import AppTest

def test_streamlit_app_renders_without_exceptions():
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception, f"App encountered exception: {at.exception}"
