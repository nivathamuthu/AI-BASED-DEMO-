from django.urls import path

from . import views

app_name = "ingest"

urlpatterns = [
    path("", views.index, name="index"),
    path("<int:pk>/", views.detail, name="detail"),
    path("<int:pk>/reprocess/", views.reprocess, name="reprocess"),
]
