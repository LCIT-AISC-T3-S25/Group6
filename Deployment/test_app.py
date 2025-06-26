# Deployment/test_app.py
from Deployment import app

def test_dummy():
    assert hasattr(app, '__file__')
