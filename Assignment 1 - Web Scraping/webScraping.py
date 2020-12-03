import bs4
from  urllib.request import urlopen 
from bs4 import BeautifulSoup as soup
import csv
import ssl
import pandas as pda

mydict = {
    'India': ['kolkata','bangalore','indore'],
    'Vietnam': ['ho-chi-minh-city','hanoi'],
    'Nepal' : ['kathmandu'],
    'Australia': ['sydney','hobart','melbourne']
}

cntxt = ssl.create_default_context()
cntxt.check_hostname = False
cntxt.verify_mode = ssl.CERT_NONE

mydata = pda.DataFrame(columns=['HotelName','City','Country','Star Rating','HotelPrice','Amenities','Hotel Description'])

for country in mydict:
    for city in mydict[country]:
        myURL = "https://www.goibibo.com/hotels/hotels-in-"+city+"-ct/";
        myHTML = urlopen(myURL,context=cntxt).read()
        myParser = soup(myHTML,"html.parser")
        myDataSet = pda.DataFrame(columns=['HotelName','City','Country','Star Rating','HotelPrice','Amenities','Hotel Description'])
        myDataSet['HotelName']= [tag.contents[0] for tag in myParser.findAll("a",{"itemprop": "name"})]
        myDataSet['City']= city
        myDataSet['Country']= country
        myDataSet['Star Rating']= [tag.contents[0].split(' / ')[0] for tag in myParser.findAll("span",{"itemprop": "ratingValue"})]
        myDataSet['HotelPrice']= [tag.contents[0] for tag in myParser.findAll("p",{"itemprop": "priceRange"})]
        myDataSet['Amenities']= pda.Series([tag.contents for tag in myParser.findAll("p",{"class": "AmenitiesListstyles__TextWrapper-sc-19dqtu1-7 kQzGvm"})])
        myDataSet['Hotel Description']=pda.Series([tag.contents for tag in myParser.findAll("span",{"class": "HotelCardstyles__RoomTypeTextWrapper-sc-1s80tyk-14 lkvAhT"})])
        mydata = mydata.append(myDataSet)
mydata.to_csv('GoIbibo.csv', index = False)