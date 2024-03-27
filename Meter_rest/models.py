from django.db import models

class Data(models.Model):
    meter = models.CharField(max_length=255)
    qr = models.CharField(max_length=255, null=True, blank=True)
    time_create = models.DateTimeField(auto_now_add=True)
    objects = models.Manager()

    def __repr__(self):
        return f"{self.meter} - счетчик. {self.qr} - QR-код"

class ImageData(models.Model):
    image = models.ImageField(upload_to='WFCR_user/images/')
    image_path = models.CharField(max_length=255, blank=True, null=True)

    def save(self, *args, **kwargs):
        self.image_path = self.image.url
        super(ImageData, self).save(*args, **kwargs)

    def __str__(self):
        return (f"{self.image}")