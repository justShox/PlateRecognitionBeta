# import re
#
#
#
# # Initialize the OCR reader if needed
# # ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
#
#
# # Define position mappings for the formats
# digit_positions_format1 = [0, 1, 3, 4, 5]
# char_positions_format1 = [2, 6, 7]
#
# digit_positions_format2 = [0, 1, 2, 3, 4]
# char_positions_format2 = [5, 6, 7]
#
# digit_positions_format3 = [0, 1, 2, 3, 4, 5]
# char_positions_format3 = [6, 7]
#
#
# # Mapping dictionaries for character conversion
# dict_char_to_int = {'O': '0',
#                     'I': '1',
#                     'B': '3',
#                     'A': '4',
#                     'G': '6',
#                     'Z': '7',
#                     'S': '5'}
#
# dict_int_to_char = {'0': 'O',
#                     '1': 'I',
#                     '3': 'B',
#                     '8': 'B',
#                     '4': 'A',
#                     '6': 'G',
#                     '7': 'Z',
#                     '5': 'S'}
#
# region_change = {'61': '01',
#                  'G1': '01',
#                  'GI': '01',
#                  'D1': '01',
#                  '07': '01',
#                  'PD': '90',
#                  'PG': '90',
#                  'DG': '90',
#                  'JO': '10',
#                  'J0': '10',
#                  '1O': '10',
#                  'L4': '40',
#                  'L0': '40',
#                  'LO': '40',
#                  'LD': '40',
#                  'AD': '40',
#                  'LL': '40',
#                  'LG': '40',
#                  '4O': '40',
#                  '4G': '40',
#                  '4D': '40',
#                  'G0': '60',
#                  'GO': '60',
#                  '6O': '60',
#                  '6G': '60',
#                  'FS': '75',
#                  }
#
#
# regions = ["01", "10", "20", "30", "40", "50", "25", "60", "70", "75", "80", "85", "90", "95"]
#
#
# def clean_plate_text(text):
#     """Removing non-alphanumeric characters."""
#     return re.sub(r'[^A-Z0-9]', '', text)
#
#
# def correct_region(text):
#     region = text[:2]
#     rest_of_text = text[2:]
#
#     # Check if region needs to be replaced
#     if region in region_change:
#         region = region_change[region]
#         text = region + rest_of_text
#
#     return text
#
#
# def license_complies_format(text):
#     """
#     Check if the license plate text complies with the required formats:
#     - "11 A 111 AA"
#     - "11 111 AAA"
#
#     2 line - "11 111 AAA"
#     2 line - "11 1111 AA"
#
#     Args:
#         text (str): License plate text.
#
#     Returns:
#         bool: True if the license plate complies with one of the formats, False otherwise.
#     """
#     # Remove any spaces from the text
#     text = text.replace(' ', '')
#
#     # Define the regular expressions for the two formats
#     format1 = re.compile(r'^\d{2}[A-Z]\d{3}[A-Z]{2}$') #"11 A 111 AA" https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_shakhsi-UZ.png
#     format2 = re.compile(r'^\d{2}\d{3}[A-Z]{3}$') #"11 111 AAA" https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_dolati2-UZ.png  https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_dolati-UZ.png
#     format3 = re.compile(r'^\d{2}\d{4}[A-Z]{2}$') #"11 1111 AA" https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_tereyli-UZ.png
#     format4 = re.compile(r'^\d{2}\d{3}[A-Z]{2}$') #"11 111 AA" https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_motor-UZ.png
#     format5 = re.compile(r'^\d{2}[MH]\d{6}$') #"11 A 111111" https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_khareji-UZ.png  https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_khareji2-UZ.png https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_mohajer-UZ.png  https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_mohajer2-UZ.png
#     format6 = re.compile(r'^UN\d{4}$') #"UN 1111" https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_SMM-UZ.png  https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan#/media/File:Pelak_SMM2-UZ.png
#
#     # Check if the text matches either format
#     if format1.match(text) or format2.match(text) or format3.match(text) or format4.match(text) or format5.match(text) or format6.match(text):
#         if text[:2] in regions:
#             return True
#     return False
#
#
# # def format_license(text):
# #     """
# #     Format the license plate text by converting characters according to their positions
# #     and mapping dictionaries.
#
# #     Args:
# #         text (str): License plate text.
#
# #     Returns:
# #         str: Formatted license plate text.
# #     """
# #     # Remove spaces from the text for processing
# #     text = text.replace(' ', '')
#
# #     text = correct_region(text)
#
# #     formatted_text = ''
#
# #     # Check and format based on the lengths and possible formats
# #     if len(text) == 8:
# #         # Check for format "11 A 111 AA"
# #         if (text[0].isdigit() and text[1].isdigit() and
# #             text[2].isalpha() and text[3].isdigit() and
# #             text[4].isdigit() and text[5].isdigit() and
# #             text[6].isalpha() and text[7].isalpha()):
# #             mapping = {
# #                 'digit': dict_char_to_int,
# #                 'char': dict_int_to_char
# #             }
# #             for i in range(8):
# #                 if i in digit_positions_format1 and text[i].isalpha():
# #                     formatted_text += mapping['digit'].get(text[i], text[i])
# #                 elif i in char_positions_format1 and text[i].isdigit():
# #                     formatted_text += mapping['char'].get(text[i], text[i])
# #                 else:
# #                     formatted_text += text[i]
#
# #         # Check for format "11 111 AAA"
# #         elif (text[0].isdigit() and text[1].isdigit() and
# #               text[2].isdigit() and text[3].isdigit() and
# #               text[4].isdigit() and text[5].isalpha() and
# #               text[6].isalpha() and text[7].isalpha()):
#
# #             mapping = {
# #                 'digit': dict_char_to_int,
# #                 'char': dict_int_to_char
# #             }
# #             for i in range(8):
# #                 if i in digit_positions_format2 and text[i].isalpha():
# #                     formatted_text += mapping['digit'].get(text[i], text[i])
# #                 elif i in char_positions_format2 and text[i].isdigit():
# #                     formatted_text += mapping['char'].get(text[i], text[i])
# #                 else:
# #                     formatted_text += text[i]
#
# #         # Check for format "11 1111 AA"
# #         elif (text[0].isdigit() and text[1].isdigit() and
# #               text[2].isdigit() and text[3].isdigit() and
# #               text[4].isdigit() and text[5].isdigit() and
# #               text[6].isalpha() and text[7].isalpha()):
#
# #             mapping = {
# #                 'digit': dict_char_to_int,
# #                 'char': dict_int_to_char
# #             }
# #             for i in range(8):
# #                 if i in digit_positions_format3 and text[i].isalpha():
# #                     formatted_text += mapping['digit'].get(text[i], text[i])
# #                 elif i in char_positions_format3 and text[i].isdigit():
# #                     formatted_text += mapping['char'].get(text[i], text[i])
# #                 else:
# #                     formatted_text += text[i]
# #         else:
# #             formatted_text = text
# #     return formatted_text
#
#
# def reformat(text):
#
#     formatted_text = ''
#
#     if len(text) == 8:
#         mapping = {
#             'digit': dict_char_to_int,
#             'char': dict_int_to_char
#         }
#         for i in range(8):
#             if i in digit_positions_format1 and text[i].isalpha():
#                 formatted_text += mapping['digit'].get(text[i], text[i])
#             elif i in char_positions_format1 and text[i].isdigit():
#                 formatted_text += mapping['char'].get(text[i], text[i])
#             else:
#                 formatted_text += text[i]
#
#         if not license_complies_format(formatted_text):
#             formatted_text = ''
#             for i in range(8):
#                 if i in digit_positions_format2 and text[i].isalpha():
#                     formatted_text += mapping['digit'].get(text[i], text[i])
#                 elif i in char_positions_format2 and text[i].isdigit():
#                     formatted_text += mapping['char'].get(text[i], text[i])
#                 else:
#                     formatted_text += text[i]
#
#         return formatted_text
#
#     return text
#
#
# def read_license_plate(license_plate_crop, ocr):
#
#     detections = ocr.ocr(license_plate_crop)
#
#     if detections[0]:
#         text, score = '', 0
#         for detection in detections[0]:
#
#             text += detection[1][0]
#             score += detection[1][1]
#
#         score = score / len(detections[0])
#
#         text = clean_plate_text(text)
#
#         text = text.upper().replace(' ', '')
#
#         text = text[:9]
#
#         if not len(detections[0]) in (1, 2):
#             return None, None
#
#         if len(detections[0]) == 2:
#
#             if len(text) == 7:
#                 text = text[3:5] + text[:3] + text[5:7]
#             elif len(text) == 8:
#                 text = text[-2:] + text[:-2]
#
#         print(text, score)
#
#         text = correct_region(text)
#
#         if len(text) == 8:
#
#             formatted = text
#
#             if license_complies_format(text):
#                 return formatted, score
#
#             formatted = reformat(text)
#
#             if license_complies_format(formatted):
#                 return formatted, score
#
#         elif len(text) == 7 or len(text) == 9:
#
#             if license_complies_format(text):
#                 return text, score
#
#     return None, None
#
# # https://en.wikipedia.org/wiki/Vehicle_registration_plates_of_Uzbekistan


import re
from paddleocr import PaddleOCR

# Initialize the OCR reader if needed
# ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Define position mappings for the formats
digit_positions_format1 = [0, 1, 3, 4, 5]
char_positions_format1 = [2, 6, 7]

digit_positions_format2 = [0, 1, 2, 3, 4]
char_positions_format2 = [5, 6, 7]

digit_positions_format3 = [0, 1, 2, 3, 4, 5]
char_positions_format3 = [6, 7]

# Mapping dictionaries for character conversion
dict_char_to_int = {
    'O': '0',
    'I': '1',
    'B': '3',
    'A': '4',
    'G': '6',
    'Z': '7',
    'S': '5'
}

dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'B',
    '8': 'B',
    '4': 'A',
    '6': 'G',
    '7': 'Z',
    '5': 'S'
}

region_change = {
    '61': '01',
    'G1': '01',
    'GI': '01',
    'D1': '01',
    '07': '01',
    'PD': '90',
    'PG': '90',
    'DG': '90',
    'JO': '10',
    'J0': '10',
    '1O': '10',
    'L4': '40',
    'L0': '40',
    'LO': '40',
    'LD': '40',
    'AD': '40',
    'LL': '40',
    'LG': '40',
    '4O': '40',
    '4G': '40',
    '4D': '40',
    'G0': '60',
    'GO': '60',
    '6O': '60',
    '6G': '60',
    'FS': '75',
}

regions = ["01", "10", "20", "30", "40", "50", "25", "60", "70", "75", "80", "85", "90", "95"]

def clean_plate_text(text):
    """Remove non-alphanumeric characters from the text."""
    return re.sub(r'[^A-Z0-9]', '', text)

def correct_region(text):
    """Correct the region code in the license plate text."""
    region = text[:2]
    rest_of_text = text[2:]

    # Check if region needs to be replaced
    if region in region_change:
        region = region_change[region]
        text = region + rest_of_text

    return text

def license_complies_format(text):
    """
    Check if the license plate text complies with the required formats:
    - "11 A 111 AA"
    - "11 111 AAA"
    - "11 1111 AA"
    - "11 111 AA"
    - "11 A 111111"
    - "UN 1111"

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with one of the formats, False otherwise.
    """
    # Remove any spaces from the text
    text = text.replace(' ', '')

    # Define the regular expressions for the formats
    format1 = re.compile(r'^\d{2}[A-Z]\d{3}[A-Z]{2}$')  # "11 A 111 AA"
    format2 = re.compile(r'^\d{2}\d{3}[A-Z]{3}$')  # "11 111 AAA"
    format3 = re.compile(r'^\d{2}\d{4}[A-Z]{2}$')  # "11 1111 AA"
    format4 = re.compile(r'^\d{2}\d{3}[A-Z]{2}$')  # "11 111 AA"
    format5 = re.compile(r'^\d{2}[MH]\d{6}$')  # "11 A 111111"
    format6 = re.compile(r'^UN\d{4}$')  # "UN 1111"

    # Check if the text matches any of the formats
    if (format1.match(text) or format2.match(text) or format3.match(text) or
        format4.match(text) or format5.match(text) or format6.match(text)):
        if text[:2] in regions:
            return True
    return False

def reformat(text):
    """Reformat the license plate text based on predefined rules."""
    formatted_text = ''

    if len(text) == 8:
        mapping = {
            'digit': dict_char_to_int,
            'char': dict_int_to_char
        }
        for i in range(8):
            if i in digit_positions_format1 and text[i].isalpha():
                formatted_text += mapping['digit'].get(text[i], text[i])
            elif i in char_positions_format1 and text[i].isdigit():
                formatted_text += mapping['char'].get(text[i], text[i])
            else:
                formatted_text += text[i]

        if not license_complies_format(formatted_text):
            formatted_text = ''
            for i in range(8):
                if i in digit_positions_format2 and text[i].isalpha():
                    formatted_text += mapping['digit'].get(text[i], text[i])
                elif i in char_positions_format2 and text[i].isdigit():
                    formatted_text += mapping['char'].get(text[i], text[i])
                else:
                    formatted_text += text[i]

        return formatted_text

    return text

def read_license_plate(license_plate_crop, ocr):
    """
    Read and process the license plate text from a cropped image.

    Args:
        license_plate_crop (numpy.ndarray): Cropped image of the license plate.
        ocr (PaddleOCR): Initialized PaddleOCR object.

    Returns:
        tuple: (formatted_text, score) if successful, (None, None) otherwise.
    """
    detections = ocr.ocr(license_plate_crop)

    if detections[0]:
        text, score = '', 0
        for detection in detections[0]:
            text += detection[1][0]
            score += detection[1][1]

        score = score / len(detections[0])

        text = clean_plate_text(text)
        text = text.upper().replace(' ', '')
        text = text[:9]

        if not len(detections[0]) in (1, 2):
            return None, None

        if len(detections[0]) == 2:
            if len(text) == 7:
                text = text[3:5] + text[:3] + text[5:7]
            elif len(text) == 8:
                text = text[-2:] + text[:-2]

        print(text, score)

        text = correct_region(text)

        if len(text) == 8:
            formatted = text

            if license_complies_format(text):
                return formatted, score

            formatted = reformat(text)

            if license_complies_format(formatted):
                return formatted, score

        elif len(text) == 7 or len(text) == 9:
            if license_complies_format(text):
                return text, score

    return None, None