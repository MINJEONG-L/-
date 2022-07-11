# 머신러닝

## 빅데이터 분석절차
  >> 기획:목적 - 필요한 데이터 수집(Data Collection) - 데이터 전처리(Data Preprocessing) - 모델 선택(Model Selection) - 평가 및 적용(Evaluation & Application)
  >> 보통 95%가 넘어야 모델 상용화 가능 : 평가 및 적용에서 

* titanic data 속성  
![image](https://user-images.githubusercontent.com/82145878/178171950-0780e7e9-a037-4468-b804-6cd022e4a0b4.png)  

  >> 맞출 것을 종속변수에 넣음
  >> 2개까지 분류 : 이진분류 __그 이상은 다중분류__  
  >> 연속된 값의 근사값을 맞추고자 할 때 : 회귀분석  
  >> sibsp : 형제 자매가 같이 탔는지  
  >> parch : 가족이 같이 탔는지   그 외 alone 변수도있음  
  >> embarked : 최초 어디 항에서 탔느냐  
  >> deck : 방 번호  
  >> alive : survived 의 텍스트형  

* 특정 속성으로 분류를 시작하려면 뭐부터 봐야할까?  
   - 값들과 빈도를 본다 value_counts()  
   - ![image](https://user-images.githubusercontent.com/82145878/178172214-56367c03-a3b0-4944-b2a2-c2746307a5ce.png)  

   - 수치를 그래프로 보려면 plot()함수 사용 value_counts().plot() (내림차순)  
   - ![image](https://user-images.githubusercontent.com/82145878/178172308-55f7b43a-d529-4e41-8a0e-790bff54adb0.png)  
   - info 확인하기  
   - ![image](https://user-images.githubusercontent.com/82145878/178172405-52af5467-158d-4d4c-a4d9-49c6a2957b3c.png)  
   - 연속형 데이터 타입 : 나이, 가격,   
   - ![image](https://user-images.githubusercontent.com/82145878/178172772-dfe75f26-e081-4700-8508-2beb3767e766.png)  
   - 전체적인 형태 보기  
   - ![image](https://user-images.githubusercontent.com/82145878/178173383-00c97e24-6857-43e7-a8e5-39610446a5ab.png)  
   - ![image](https://user-images.githubusercontent.com/82145878/178175534-984b7d75-f871-4593-867a-8c6a77dba8d5.png)  
   -    ==>  왜 여섯개만 나올까 : 숫자형 데이터만 나옴! 텍스트 데이터는 계산을 할 수 없으니까  
   -  ![image](https://user-images.githubusercontent.com/82145878/178175958-1132c418-1840-409b-b441-cad1f4ac0c38.png)  
   -  







  
