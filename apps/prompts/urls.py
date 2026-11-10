from django.urls import path

from . import views

app_name = "prompts"

urlpatterns = [
    path("", views.compare, name="compare"),
]
