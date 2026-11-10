"""The recording landing page and the results page."""
import pytest
from django.urls import reverse

pytestmark = pytest.mark.django_db


def test_home_lists_ready_modules_and_marks_six_and_seven_planned(client):
    page = client.get(reverse("home")).content.decode()
    assert page.count("planned") == 2
    for name in ("Ingest", "Prompts", "RAG", "Graph", "Rules", "Agent"):
        assert name in page
    assert "Q&amp;A" in page or "Q&A" in page
    assert reverse("ingest:index") in page
    assert reverse("agents:index") in page
    assert reverse("results") in page


def test_results_page_renders_with_no_evals(client):
    page = client.get(reverse("results")).content.decode()
    assert page.count("No eval yet.") == 5
