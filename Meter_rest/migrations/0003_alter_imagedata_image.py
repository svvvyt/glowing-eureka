# Generated by Django 5.0.3 on 2024-03-24 17:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Meter_rest', '0002_imagedata_alter_data_qr'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagedata',
            name='image',
            field=models.ImageField(upload_to='Meter_rest/images/'),
        ),
    ]
