#-*- coding:utf-8 -*-
import urllib3
import json
import base64
import json
import base64
import io
import os
class ETRI_STT:
    def run(self,file_name='default'):
        openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
        accessKey = "b9e052c1-47b8-4436-b73f-078045717d2b"
        languageCode = "korean"
        file_names = file_name
        ip = file_name.split('_')[0]
        file_name = os.path.join(os.path.dirname(__file__), "wav", file_name)

        # Loads the audio into memory
        file = open(file_name, "rb")
        audioContents = base64.b64encode(file.read()).decode("utf8")
        file.close()

        requestJson = {
            "access_key": accessKey,
            "argument": {
                "language_code": languageCode,
                "audio": audioContents
            }
        }
 	
        http = urllib3.PoolManager()
        response = http.request(
            "POST",
            openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8"},
            body=json.dumps(requestJson)
        ) 

        print(response.data['result'])
        text = " "

        print("\n")
        print("--Transcript--")	
        data = json.loads(response.data.decode("utf-8", errors='ignore'))
        text=data['return_object']['recognized']
        
        print(text)
        
        return ip, text

    