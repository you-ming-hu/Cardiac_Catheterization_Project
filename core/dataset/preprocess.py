import cv2

def PreprocessParser(func_str):
    return eval(func_str)

def GrayImage(image):
    if image.shape[1:] != (512,512):
        image = cv2.resize(image,(512,512))
    if len(image.shape) == 2:
        return image
    else:
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

def RGBImage(image):
    if image.shape[1:] != (512,512):
        image = cv2.resize(image,(512,512))
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
def BGRImage(image):
    if image.shape[1:] != (512,512):
        image = cv2.resize(image,(512,512))
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        return image