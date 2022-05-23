from datetime import date
from requests import request
import json
import requests


year = date.today().year
url = f'http://api.themoviedb.org/3/discover/movie?api_key=9ae91122e24d51e65bc513b3218691dc&primary_release_year={year}&sort_by=popularity.desc'

genres = json.loads(requests.get("https://api.themoviedb.org/3/genre/movie/list?api_key=9ae91122e24d51e65bc513b3218691dc&language=en-US").text)

genre_id = f'https://api.themoviedb.org/3/discover/movie?api_key=9ae91122e24d51e65bc513b3218691dc&with_genres=28&sort_by=popularity.desc'

data = requests.get(url)
data_1 = json.loads(requests.get(genre_id).text)

print(data_1)