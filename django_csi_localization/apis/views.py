from django.http import JsonResponse
from django.shortcuts import render
from django.core import files
from django.core.files.storage import default_storage, FileSystemStorage
from rest_framework.decorators import api_view
from apis.utils import scan_environment
from apis.yolo_utils import detect, gen_depth_map, calc_density
from rest_framework.decorators import parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
import timeit
from PIL import Image
from django.core.files import File
# Create your views here.
@api_view(('POST',))
@parser_classes([MultiPartParser, FormParser])
def environment(request):
    #print(request.data))
    start = timeit.default_timer()
    back_img = request.data['back']
    front_img = request.data['front']
    imgs = [back_img.temporary_file_path(), front_img.temporary_file_path()]
    W, H, L, V = scan_environment(imgs)
    stop = timeit.default_timer()
    print("runtime: " + str(stop - start))
    return JsonResponse({'W': W, 'H': H, 'L': L, 'V': V})

@api_view(('POST',))
@parser_classes([MultiPartParser, FormParser])
def density(request):
  start = timeit.default_timer()
  back_img = request.FILES['back']
  front_img = request.FILES['front']
  fs = FileSystemStorage(location='density/')
  back = fs.save(back_img.name, back_img)
  front = fs.save(front_img.name, front_img)
  print(back)
  print(front)
#  imgs = [back_img.temporary_file_path(), front_img.temporary_file_path()]
#  file_name = 'back.png'
#  img = Image.open(back_img)
#  temp_file = files.File(img, name=file_name)
#  path = default_storage.save('./density', temp_file)

#  for img in imgs:
#    density += calc_density(img)
  density = calc_density(['./density/' + back, './density/' +  front])
  stop = timeit.default_timer()
  print("runtim: " + str(stop - start))
  print(density)
  return JsonResponse({'D': density})
