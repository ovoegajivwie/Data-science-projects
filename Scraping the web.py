import requests
from bs4 import BeautifulSoup

print('Get the cource code for any website of your choice simply by inputin the url')
input_url = input()
url = input_url
r = requests.get(url)
t = r.text
soup = BeautifulSoup(t)
print(soup.prettify())
