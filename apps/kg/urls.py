from django.urls import path

from apps.kg import views

app_name = "kg"

urlpatterns = [
    path("", views.index, name="index"),
    path("graph/", views.graph_frame, name="graph"),
    path("node/", views.node_detail, name="node"),
]
