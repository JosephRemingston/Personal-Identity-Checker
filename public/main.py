from detector import detect_ids
from PersonalIdentityIdentification import PersonalIdentityIdentification
input_image_path = "C:/object_detection/IMG_1760.jpeg"
if PersonalIdentityIdentification(input_image_path) == True:
    detect_ids(input_image_path)
else:
    print("No Personal Identity Information given")