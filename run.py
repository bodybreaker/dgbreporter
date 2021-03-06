import os
import time
from glob import glob
from etri_stt import ETRI_STT
import keywords
from pydub import AudioSegment
import math
import natsort
import time
from flask import Flask,request
from werkzeug.utils import secure_filename

###  원 파일을 1분 이하로 나누는 함수 ### 
class SplitWavAudioMubin():    
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 18 * 1000
        t2 = to_min * 18 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export("./wav/"+ split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 18)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(0) + '_' + str(i) + ".wav"
            #split_fn = str(i)+".wav"
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')
class watchdog:
    def __init__(self,fileName):
        self.ipDict = dict()
        self.FileList = list()
        self.checkCount = 0
        self.fileName = fileName
    def run(self):
        folder = './'
        #file = 'YTN_bong_3.wav'
        file = self.fileName
        split_wav = SplitWavAudioMubin(folder, file)                                 
        split_wav.multiple_split(min_per_split=1)			## 전체 wav파일을 1분 이하의 파일들로 분리함 
        #self.FileList = sorted(glob("./wav/*"))     ## 분리된 wav 파일을 glob으로 읽음
        lst = os.listdir("./wav/")
        self.FileList = natsort.natsorted(lst)
        print(self.FileList)
        ETRI= ETRI_STT()					## ETRI stt를 활용하여 읽음 
        print("Start...")
        NewFileList = self.FileList			


        for fileListName in NewFileList:				
            self.checkCount = 0
            ip, text = ETRI.run(fileListName.split("/")[-1])		## text  : stt 결과 spring형태의 text 결과가 저장된 변수값
            
            if ip in self.ipDict:				## ip(interpretation percent) : 요약과 keyword 추출시, 각 문장단위로 앞 문장과의 연관성과 전체 문장내에서의 해당 문장의 영향도를 분석하기 위한 변수 
                self.ipDict[ip] += text
            else:
                self.ipDict[ip] = text

        # 완료된 파일 삭제
        for fileListName in NewFileList:
            print('파일삭제 >> ',fileListName)
            os.remove('./wav/'+fileListName)

        #time.sleep(5) 
        self.checkCount += 1
        text = " "

        text += self.ipDict['0']
        #이부분 핖요하지않읋거같은데?
        #for key in self.ipDict.keys():				## kss NLP kaggle library 에서 제공하는 .key() 함수를 통해 문장을 단어단위로 분류함.  
        #    text += self.ipDict[key]

        keywords.extract(text)				## stt 결과값인 text 변수를 요약 알고리즘 (keywords.py) 함수로 실행 후, 요약된 문단과 keyword 단어 리턴 호출함.
        return text

app = Flask(__name__)

@app.route('/test')
def user():
    WATCHDOG = watchdog('short.wav')
    text = WATCHDOG.run()
    os.system('echo "회의록 요약본 첨부" | mail -s "[DGB Reporter] 회의 요약결과" bodybreaker@naver.com -A ./word/minutes_result.docx')
    return text

@app.route('/stt',methods=['POST'])
def upload_file():
    f = request.files['file']
    f.save(secure_filename(f.filename))
    WATCHDOG = watchdog(f.filename)
    text = WATCHDOG.run()
    os.system('echo "회의록 요약본 첨부" | mail -s "[DGB Reporter] 회의 요약결과" bodybreaker@naver.com,eric.yun8911@gmail.com -A ./word/minutes_result.docx')

    return text


if __name__ == "__main__":
    #WATCHDOG = watchdog()
    #WATCHDOG.run()
    
    app.run(debug=True)

    #os.startfile("./word/minutes.docx")   ##word와 연동 하는 코드 


