import os
import sys
from flask import Flask

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import app

print("Starting Crop Disease Detection Web Application...")
print("Visit http://localhost:5000 in your browser")

if __name__ == '__main__':
    # Set host to '0.0.0.0' to allow external connections
    # Set port to 5000 explicitly
    app.run(host='0.0.0.0', port=5000, debug=True) 