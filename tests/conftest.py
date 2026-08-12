"""Force mock investigators during automated tests (no live network)."""

import os

import pytest


@pytest.fixture(autouse=True)
def _force_mock_for_tests(monkeypatch):
    monkeypatch.setenv("USE_MOCK_MODEL", "true")
    monkeypatch.setenv("SIGNAL_SENTRY_FORCE_MOCK", "true")
    monkeypatch.setenv("MODEL_API_KEY", "nemoclaw-local-placeholder")
