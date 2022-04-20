import requests
from urllib import parse
import xmltodict
import json

'''
url = # api url 주소
service_key = # 서비스 키
params = {
    
}
'''

response = requests.got()
# print(response.content)

dict_result = json.dumps(xmltodict.parse(response.content), indent = 4)
print(dict_result)