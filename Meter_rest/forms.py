from django import forms
from .models import ImageData


class ImageForm(forms.ModelForm):
    class Meta:
        model = ImageData
        fields = ['image']
