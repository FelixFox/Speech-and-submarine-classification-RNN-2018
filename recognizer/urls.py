from django.urls import path
from . import views


urlpatterns = [
    path("", views.index, name="index"),
    path("submarine/", views.submarine, name="recognize-submarine"),
    path("speech/", views.speech, name="recognize-speech"),
    path("history/", views.history, name="history")
]
