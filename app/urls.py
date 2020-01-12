from django.urls import path
from .views import train,test

urlpatterns = [
    path('',train, name='train'),
    path('train',train, name='train'),
    path('test',test, name='test'),
]
