import cv2
import pytesseract

def getText (box,painting):

    extractedText = ""

    if(box[3]>1 and box[2]>1):
        textField = painting [box[1]:box[3],box[0]:box[2]]

        textGray = cv2.cvtColor(textField, cv2.COLOR_BGR2GRAY)
        th, textFieldBin = cv2.threshold(textGray, 0,255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # cv2.imshow("text",textFieldBin)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        # pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
                                                
        extractedText = pytesseract.image_to_string(textFieldBin, lang='cat+spa+eng')
    
    print(f'Text in bounding box: {extractedText}')

    return extractedText