from rest_framework import serializers
from .models import Data, ImageData


class DataSerializer(serializers.ModelSerializer):
    class Meta:
        model = Data
        fields = "__all__"


class ImageDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageData
        fields = "__all__"
