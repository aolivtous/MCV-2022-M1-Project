# Team5
- Anna Oliveras
- Guillem Capellera
- Marcos Frías
- Àngel Herrero


### Instalation
- Create a new environment (Highly recommended)
- Use Python 3.9.13
- Install dependencies with: *pip install -r requirements.txt*

### Needed folders organization
Top level directory layout\
├── Team5&nbsp;&nbsp;&nbsp;&nbsp;&rarr; Our repository with the code\
├── db&nbsp;&nbsp;&nbsp;&nbsp;&rarr; Folder with the database\
├── qsd1_w1&nbsp;&nbsp;&nbsp;&nbsp;&rarr; Folder with the query1 images\
├── qsd2_w2&nbsp;&nbsp;&nbsp;&nbsp;&rarr; Folder with the query2 images and masks

### Root file
0. **py** &rarr; *python py (args)* to execute it
### Arguments:
1. **First:** Query name &rarr; (qsd1_w1, qsd2_w1, qst1_w1, qst2_w1)
2. **Second:** Method to search most similar painting &rarr; (1, 2)
3. **Third:** Method to generate the masks &rarr; (1, 2)
4. **Four:** Images have backgrounds &rarr; (True, False)
5. **Five:** Images have text box &rarr; (True,False)
6. **Six:** Images may have two paintings &rarr; (True, False)
7. **Seven:** Solutions are available (to compute scores) &rarr; (True, False)

Boolean values can be defined in several ways:
- **True** &rarr; {1, 't', 'true', 'yes'}
- **False** &rarr; else

#### Examples:
 - *python py qsd2_w1 1 2 True True*:
   - Query 1 with method 1 to search the painting, method 2 to generate the masks, images with background and solutions available to calculate score.
 - *python py qst1_w2 1 1 no no*:
   - Test query 2 with method 1 to search the painting, method 1 to generate the masks, images without background and no solutions available to calculate score.

**Important:** It's mandatory to define all the arguments even when they are not needed. (i.e when searching for coincidences with images without background it's necessary to include a mask method even though it's not used)

   