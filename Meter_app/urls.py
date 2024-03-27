from django.conf import settings
from django.conf.urls.static import static

from django.urls import path, include
from rest_framework import routers
from Meter_rest.views import *


class CustomRouter(routers.SimpleRouter):
    routes = [
        routers.Route(
            url=r'^{prefix}/$',
            mapping={'get': 'list', 'post': 'create'},
            name='{basename}-list',
            detail=False,
            initkwargs={'suffix': 'List'}
        ),
        routers.Route(
            url=r'^{prefix}/{lookup}$',
            mapping={'get': 'retrieve', 'put': 'update', 'delete': 'destroy'},
            name='{basename}-detail',
            detail=True,
            initkwargs={'suffix': 'Detail'}
        ),
    ]


router = CustomRouter()
router.register(r'data', DataViewSet, basename='data')

urlpatterns = [
    path('api/', include(router.urls)),  # Запятая добавлена здесь
    path('upload-image/', upload_image, name='upload_image'),
    path('api/images/', ImageView.as_view(), name='image-list'),
    path('api/images/<int:image_id>/', image_view, name='image-detail'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

