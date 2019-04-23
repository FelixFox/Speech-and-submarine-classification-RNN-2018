import os
import uuid

from django.http import HttpResponseRedirect, JsonResponse, HttpResponse
from django.shortcuts import render

from .services.submarine.recognition import SubmarineRecognizer
from .services.speech.recognition import SpeechRecognizer

from .forms import UploadFileForm
from .models import Record

from cgi import escape
from xhtml2pdf import pisa
from io import BytesIO, StringIO
from django.template.loader import get_template
from django.template import Context
from django.core import serializers


# Create your views here.


def index(request):
    return render(request, "index.html")


def save_record_to_db(recognize_result, rec_type, user, filename, uuid_filename):
    item = Record()
    if user.is_authenticated:
        item.user = user
    item.rec_type = rec_type
    item.rec_score = recognize_result["score"]
    item.rec_class = recognize_result["class"]
    item.sample_rate = recognize_result["sample_rate"]
    item.nn_name = recognize_result["nn_name"]
    item.filename = filename
    item.uuid_filename = uuid_filename
    item.save()


def submarine(request):
    if request.method == 'POST':

        local_file_path = handle_uploaded_file(request.FILES['sound'])

        # TODO: Single instance per app
        recognizer = SubmarineRecognizer(recorded_file=local_file_path)
        prediction = recognizer.predict()
        recognize_result = get_submarine_recognize_result(
            recognizer, prediction)
        os.remove(local_file_path)
        print(request.user)
        save_record_to_db(recognize_result, "submarine", request.user,
                          request.FILES['sound'].name,
                          os.path.basename(local_file_path))

        return JsonResponse(recognize_result)

    return render(request, "recognize_submarine.html")


def speech(request):

    if request.method == 'POST':
        local_file_path = handle_uploaded_file(request.FILES['sound'])

        # TODO: Single instance per app
        recognizer = SpeechRecognizer(recorded_file=local_file_path)
        prediction = recognizer.predict()
        recognize_result = get_speech_recognize_result(
            recognizer, prediction)
        os.remove(local_file_path)
        save_record_to_db(recognize_result, "speech", request.user,
                          request.FILES['sound'].name,
                          os.path.basename(local_file_path))

        return JsonResponse(recognize_result)

    return render(request, "recognize_speech.html")


# TODO: Only for authenticated users
def history(request):
    user_records = Record.objects.filter(
        user=request.user.id).order_by('-date')

    return render(request, "history.html", {"records": user_records})


def get_submarine_recognize_result(recognizer, prediction):
    freqs, amps = recognizer.wav_to_features()
    all_scores = [{"class": cl, "score": y.item()}
                  for (cl, y) in prediction["y"].items()]
    return {
        "freqs": freqs.tolist(),
        "amps": amps.tolist(),
        "nn_name": "1D Convolution NN",
        # TODO: Fix empty
        "features_shape": recognizer.f_shape,
        "class": prediction["predicted_class"],
        "score": prediction["score"].item(),
        "all_scores": all_scores,
        "sample_rate": recognizer.audio_preprocessor.get_samplerate(),
        "time_recognition": 1
    }


def get_speech_recognize_result(recognizer, prediction):
    mels = recognizer.wav_to_features()
    all_scores = [{"class": cl, "score": y.item()}
                  for (cl, y) in prediction["y"].items()]

    print(type(mels))
    return {
        "mels": mels.T.tolist(),
        "nn_name": "2D Convolution + LSTM NN",
        # TODO: Fix empty
        "features_shape": recognizer.f_shape,
        "class": prediction["predicted_class"],
        "score": prediction["score"].item(),
        "all_scores": all_scores,
        "sample_rate": recognizer.audio_preprocessor.get_samplerate(),
        "time_recognition": 1
    }


# TODO: Do not save in file, work with data in memory
def handle_uploaded_file(file):
    path = "{}.wav".format(uuid.uuid4().hex)
    with open(path, "wb+") as dest:
        for chunk in file.chunks():
            dest.write(chunk)

    return path
