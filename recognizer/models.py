from django.db import models
from django.template.defaultfilters import slugify
from django.contrib.auth.models import User


# Create your models here.

class Record(models.Model):
    REC_TYPE_CHOISES = (("speech", "Speech recognition"),
                        ("submarine", "Submarine's distanse recognition"))

    date = models.DateTimeField(auto_now_add=True)
    rec_type = models.CharField(choices=REC_TYPE_CHOISES, max_length=15)
    rec_class = models.CharField(max_length=15, default="Unknown")
    rec_score = models.DecimalField(max_digits=6, decimal_places=3)
    sample_rate = models.IntegerField()
    nn_name = models.CharField(max_length=15)
    user = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True)
    filename = models.CharField(
        max_length=255, default=None, null=True)
    uuid_filename = models.CharField(
        max_length=255, default=None, null=True)
