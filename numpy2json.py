import json
from json import JSONEncoder
import numpy as np
import requests

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def check_image(image):
    print(f'type - {type(image)}')
    print(np.shape(image))
    print(image.ndim)
    print(np.size(image))
    return 

# Test Index
# url = "http://0.0.0.0:5000/"
# response = requests.request("GET", url)
# print(response.text)

# Test POST name
# url = "http://0.0.0.0:5000/name?"
# payload={'name': 'demian'}
# response = requests.request("POST", url, data=payload )
# print(response.text)

# Test Sending Image
# url = "http://0.0.0.0:5000/myImage"
# files=[
#   ('image',('my_matrix.dat',open('/home/demian/projects/upwork/Semicon-wafer-classification/web/my_matrix.dat','rb'),'application/octet-stream'))
# ]
# response = requests.request("POST", url, data=payload, files=files)
# print(response.text)

# Test Semicon
numpyData = np.load("/home/demian/projects/upwork/Semicon-wafer-classification/web/my_matrix.dat",allow_pickle=True)

# Serialization
encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)  

# Deserialization
finalNumpyArray = np.fromstring(encodedNumpyData,np.uint8)
check_image(finalNumpyArray)

url='http://0.0.0.0:5000/predict'
headers={'user-agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36'}
r=requests.post(url,headers=headers,files={"image": encodedNumpyData})
print(r.text)














