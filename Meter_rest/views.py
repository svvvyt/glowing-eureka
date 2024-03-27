from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import mixins
from rest_framework.viewsets import GenericViewSet
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from .forms import ImageForm
from .models import Data, ImageData
from .serializers import DataSerializer, ImageDataSerializer
from rest_framework import generics
from django.conf import settings

from WFCR_user.main import create_api_data_image

import os
import glob


class DataViewSet(mixins.CreateModelMixin,
                  mixins.DestroyModelMixin,
                  mixins.RetrieveModelMixin,
                  mixins.UpdateModelMixin,
                  mixins.ListModelMixin,
                  GenericViewSet):
    queryset = Data.objects.all()
    serializer_class = DataSerializer

    def get_queryset(self):
        pk = self.kwargs.get("pk")

        if not pk:
            return Data.objects.all()

        return Data.objects.filter(pk=pk)


@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            image_path = form.save().image.path  # Get the saved image path
            # image_path_hard = fr'C:\Users\user\Downloads\true-meter\WFCR_user\images\{form.cleaned_data['image']}'
            create_api_data_image(image_path)
            form.clean()
            os.remove(image_path)
            return JsonResponse({'success': 'Image uploaded successfully'})  # Возвращаем успешный ответ в формате JSON
        else:
            return JsonResponse({'error': form.errors}, status=400)  # Возвращаем ошибки валидации формы в формате JSON
    elif request.method == 'GET':
        # Обработка GET-запроса (если необходимо)
        return JsonResponse({'message': 'GET request received'}, status=200)
    else:
        # Обработка других типов запросов
        return JsonResponse({'error': 'Method not allowed'}, status=405)


class ImageView(generics.ListCreateAPIView):
    queryset = ImageData.objects.all()
    serializer_class = ImageDataSerializer

def image_view(request, image_id):
    image = get_object_or_404(ImageData, pk=image_id)

    image_path = os.path.join(settings.MEDIA_ROOT, str(image.image))

    with open(image_path, 'rb') as img_file:
        # Читаем содержимое файла
        image_data = img_file.read()

    content_type = "image/jpeg"

    return HttpResponse(image_data, content_type=content_type)