from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from Meter_rest.models import ImageData  # Замените на вашу модель изображения


@receiver(pre_save, sender=ImageData)  # Замените на вашу модель изображения
def image_post_save(sender, instance, created, **kwargs):
    instance.image = "foto.jpg"

