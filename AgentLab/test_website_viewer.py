#!/usr/bin/env python3
"""
Simple test for the update_website_viewer function logic.
"""

import sys
from pathlib import Path

# Mock the gradio update function
class MockGradioUpdate:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __repr__(self):
        return f"gr.update({self.kwargs})"

def gr_update(**kwargs):
    return MockGradioUpdate(**kwargs)

# Mock the info object
class MockExpResult:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir

class MockInfo:
    def __init__(self, exp_result):
        self.exp_result = exp_result

# Test the logic
def test_update_website_viewer_logic():
    # Test case 1: No experiment selected
    info = MockInfo(None)
    result = update_website_viewer_logic(info, None)
    print("Test 1 (no experiment):", result)

    # Test case 2: Experiment with HTML files
    exp_result = MockExpResult("/tmp/test_exp")
    info = MockInfo(exp_result)

    # Create test HTML files
    test_dir = Path("/tmp/test_exp")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "gmail.html").write_text("<html><body>Gmail</body></html>")
    (test_dir / "calendar.html").write_text("<html><body>Calendar</body></html>")

    result = update_website_viewer_logic(info, None)
    print("Test 2 (with files):", result)

    # Test case 3: Specific file selected
    result = update_website_viewer_logic(info, "calendar.html")
    print("Test 3 (calendar selected):", result)

    # Cleanup
    import shutil
    shutil.rmtree("/tmp/test_exp")

def update_website_viewer_logic(info, selected_file=None):
    """Simplified version of the logic for testing."""
    if info.exp_result is None:
        return gr_update(choices=[], value=None), "<p>No experiment selected</p>"

    # Look for HTML files in the experiment directory
    exp_dir = Path(info.exp_result.exp_dir)
    html_files = list(exp_dir.glob("*.html"))

    if not html_files:
        return gr_update(choices=[], value=None), "<p>No generated HTML files found in this experiment</p>"

    # Create dropdown choices from HTML file names
    file_choices = [f.name for f in html_files]

    # If no file is selected, default to the first one
    if selected_file is None or selected_file not in file_choices:
        selected_file = file_choices[0]

    # Find the selected file
    html_file = next((f for f in html_files if f.name == selected_file), html_files[0])

    try:
        html_content = html_file.read_text(encoding="utf-8")
        # Use srcdoc instead of data URL for better compatibility
        iframe_html = f"""<iframe srcdoc="{html_content}"
                      style="width: 100%; height: 800px; border: none; background-color: white;">
              </iframe>"""
        return gr_update(choices=file_choices, value=selected_file), iframe_html
    except Exception as e:
        error_html = f"<p>Error loading HTML file {html_file.name}: {e}</p>"
        return gr_update(choices=file_choices, value=selected_file), error_html

if __name__ == "__main__":
    test_update_website_viewer_logic()


