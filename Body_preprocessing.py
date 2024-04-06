import json, re, nltk, csv, sys
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer
from preprocessing import Preprocessing

fclean = open('cleanedBody.txt', 'w', encoding="utf-8")
ftrain = open("Train.csv", "rt", encoding="utf-8")

s = set(stopwords.words('english'))
reader = csv.DictReader(ftrain)
i = 0
for row in reader:
    try:
        z = str(row['Body']).lower()
        f = 0
        bd = ''
        while f < len(z):
            x = z.find("<p>", f)
            y = z.find("</p>", f)
            bd = ''
            if x >= 0 and y >= 0:
                s1 = []
                s2 = []
                s3 = []
                s4 = []
                s1 = [m.start() for m in re.finditer('<code>', z[x:y])]
                s2 = [m.start() for m in re.finditer('</code>', z[x:y])]
                r = 0
                lh = 0
                while r < len(s1) and r < len(s2):
                    z = z.replace(z[s1[r] - lh:s2[r] + 7 - lh], '')
                    lh = lh + len(z[s1[r] - lh:s2[r] + 7 - lh])
                    r = r + 1
                r = 0
                y = z.find("</p>", f)
                s3 = [m.start() for m in re.finditer('<a h', z[x:y])]
                s4 = [m.start() for m in re.finditer('</a>', z[x:y])]
                lh = 0
                while r < len(s3) and r < len(s4):
                    z = z.replace(z[s3[r] - lh:s4[r] + 4 - lh], '')
                    lh = lh + len(z[s3[r] - lh:s4[r] + 4 - lh])
                    r = r + 1
                bd += str(z[x + 3:y].encode('utf8')) + ' '
                f = y + 5
            else:
                break
        words = re.sub('[!@%^&*()$:"?<>=~,;`{}|]', ' ', bd)
        words=Preprocessing(words,s)
        fclean.write(words + '\n')
        print("Row Cleaned-", row['Id'])
        i = i + 1
    except Exception as e:
        print("EXCEPTION: ", str(e))
        fclean.write('\n')
        i = i - 1
        pass

fclean.close()
ftrain.close()

print("Preprocessed the records")
print()
print("Error in the Following: ",error)
