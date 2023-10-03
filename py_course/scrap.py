import requests
import lxml
from bs4 import BeautifulSoup



def collect_authors(soup, authors):
    author_classes = soup.select(".author")
    for author in author_classes:
        authors.add(author.text)

def collect_quotes(soup, quotes):
    quotes_classes = soup.select(".text")
    for quote in quotes_classes:
        quotes.append(quote.text)

def collect_top10(soup, top10):
    top10_classes = soup.select(".tag-item")
    for top in top10_classes:
        sub_class = top.select('a')
        for sub in sub_class:
            top10.append(sub.text)

def get_page(index):
    url = base_url.format(index)
    res = requests.get(url)
    if ("No quotes found!" in res.text):
        return -1
    soup = BeautifulSoup(res.text, "lxml")
    #qt = soup.select(".col-md-8")
    #if qt[1].text.find("No quotes found!") > -1:
    #    return -1

    collect_authors(soup, authors)
    collect_quotes(soup, quotes)
    if (index == 1):
        collect_top10(soup, top10)
    return 0


authors=set()
quotes=[]
top10=[]
base_url = "http://quotes.toscrape.com/page/{}"

index = 1
result = 0

print ("Scrapping,  please wait...", end="", flush=True)
while (result == 0):
    print (".", end="", flush=True)
    result = get_page(index)
    index+=1
print (".")

for author in authors:
    print (author)
for qt in quotes:
    print (qt)
for tt in top10:
    print (tt)