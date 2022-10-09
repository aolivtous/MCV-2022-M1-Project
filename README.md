# Team5
- Create a new environment
- Use Python 3.9.13
- Install dependencies with: pip install -r requirements.txt

Top level directory layout 
├── Team5                   # Repository with the code
├── BBDD                    # Folder with the database
├── qsd1_w1                 # Folder with the query1 images
├── qsd2_w2                 # Folder with the query2 images and masks

True = 1, 't', 'true', 'yes'
False = else

ARGUMENTS:
 First  : name query = (qsd1_w1, qsd2_w1, qst1_w1, qst2_w1)
 Second : Method to search most similar painting = (1, 2)
 Third  : Method to generate mask = (1, 2)
 Four   : Images have backgrounds = (True, False)
 Five   : Solutions are available = (True, False)

 Examples:
 - qsd1_w1 1 1 False True --> query 1 with method 1 to generate the masks, method 1 to search the painting, with images       without background and with solutions available
 - qsd2_w1 1 1 True True
 - qst1_w1 1 1 False False
 - qst2_w2 1 1 True False
   