from django.urls import path

from apps.agents import views

app_name = "agents"

urlpatterns = [
    path("", views.index, name="index"),
    path("run/<int:run_id>/", views.run_detail, name="run"),
    path("run/<int:run_id>/trace/", views.run_trace, name="trace"),
    path("run/<int:run_id>/decide/", views.decide, name="decide"),
]
