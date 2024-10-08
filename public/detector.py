import cv2
import numpy as np

from regx_patterns import ID_PATTERNS

def match_template(image_path, threshold=0.4):
    image = cv2.imread(image_path, 0)
    template = cv2.imread("/home/josephremingston/code/Hackathon/gov_id_detector/image.png", 0)
    
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    print(max_val)
    return max_val >= threshold

def detect_document_type(image_path, templates):
    for doc_type, template_path in templates.items():
        if match_template(image_path, template_path):
            return str(doc_type)
    return None

def detect_ids(text):
    critical_alerts = {}
    lesser_critical_alerts = {}

    for id_type, pattern in ID_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            if id_type in ['Aadhar Card', 'Pan Card', 'Passport', 'DriverLicense', 'GSTIN', 
                           'NREGA Job Card', 'IFSC', 'VoterId_India', 'NPS_PRAN', 'CreditCard']:
                critical_alerts[id_type] = matches
            else:
                lesser_critical_alerts[id_type] = matches

    if critical_alerts:
        print("Critical Alerts Detected:")
        for id_type, matches in critical_alerts.items():
            for match in matches:
                print(f"{id_type}: {match[0]} found at position {match[1]} to {match[2]}")
    elif lesser_critical_alerts:
        print("\nLesser Critical Alerts Detected:")
        for id_type, matches in lesser_critical_alerts.items():
            for match in matches:
                print(f"{id_type}: {match[0]} found at position {match[1]} to {match[2]}")
    else:
        print("No Alerts Detected")
