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
* 결측치 처리
   - df.isnull.sum() (df.isna.sum() 과 동일)
   ![image](https://user-images.githubusercontent.com/82145878/178207078-dd56115c-211c-42a3-91f3-ce1d8404249e.png)  
   - df.isna().sum().sum() #전체 결측치 개수
   - df.drop('embark_town', axis = 1, inplace = True) 
        ==> axis 는 0이 디폴트라 행을 삭제하는데 1이면 컬럼을 삭제함 inplace는 원본 업뎃 (원본이 업뎃되면 출력 안됨)  
      __inplace는 오류안나는지 확인하고 넣기__ 
   - 결측치를 수정하는 방법 
    (1) df[df.embarked.isna()].embarked = 'S' 잘안씀  
    (2) df.embarked.fillna('S',inplace = True) 
    
   - 특정 컬럼(age)의 평균, 중앙값, 최대값 등등 찾기 df.age.describe()  
   ![image](https://user-images.githubusercontent.com/82145878/178245581-f882b6eb-f8e7-4f4c-b718-f5bdf90ef71d.png)  
    >> groupby  
    : df.groupby 특정조건을 그룹으로 묶어서 특정조건에 따라 여러 개의 데이터 프레임으로 쪼개지고 (출력은 안됨) 수정가능
    ```python
    group1 = df.groupby(by=['sex','pclass'])
    group1.age.median()
   ```  
   ![image](https://user-images.githubusercontent.com/82145878/178246269-deeed19d-1a57-4d18-9da9-c8e4dd015bba.png)  
    - unstack() 사용
    ![image](https://user-images.githubusercontent.com/82145878/178246424-970b4e4c-a26a-498f-9649-93d6b83d788a.png)  
    ```python
    df_sp_grouped.iloc[0,0] #행 0 = > 여 / 1=> 남  / 열 => pclass
    ```
    ![image](https://user-images.githubusercontent.com/82145878/178246522-554e30d1-5ea2-4b86-9f80-e4a79688e36e.png)  
    >> 주의 할 것
    df[(df.pclass == 1) & (df.sex == 'female')].age.fillna(35,inplace = True)  **:안됨**  
    df[(df.pclass == 1) & ( df.sex == 'female')&(df.age.isna())].age = f1 **:안됨**  
    ![image](https://user-images.githubusercontent.com/82145878/178190778-f882e3ca-9fd8-462c-b321-0033e1200e33.png)  
    #### loc 사용하기!
    `df.loc[조건, 바꿀 컬럼] = 값`
    ```python
    df.loc[(df.pclass == 1) & ( df.sex == 'female')&(df.age.isna()),'age'] = f1
    df.loc[(df.pclass == 2) & ( df.sex == 'female')&(df.age.isna()),'age'] = f2
    df.loc[(df.pclass == 3) & ( df.sex == 'female')&(df.age.isna()),'age'] = f3

    df.loc[(df.pclass == 1) & ( df.sex == 'male')&(df.age.isna()),'age'] = m1
    df.loc[(df.pclass == 2) & ( df.sex == 'male')&(df.age.isna()),'age'] = m2
    df.loc[(df.pclass == 3) & ( df.sex == 'male')&(df.age.isna()),'age'] = m3
    ```  
    ![image](https://user-images.githubusercontent.com/82145878/178247304-da58c85a-5c13-451a-b388-ac811ee90f7d.png)  
 
    - 이정도의 결측치 비율이면 그냥 drop!
    - deck 컬럼 자체를 버리기 `df.drop(columns = 'deck', inplace = True)`  
    
>> 오류 주의  
![image](https://user-images.githubusercontent.com/82145878/178247655-ed3a4b4c-4e39-4806-8b18-62838a342d9a.png)  
  - 왜 이런 오류가 날까?
    ==> 데이터 타입을 먼저 보자!  `df1.dtypes`
  ![image](https://user-images.githubusercontent.com/82145878/178247779-d6c8481c-3784-4977-a0ce-1a3a229ceeee.png)  
  - deck 의 데이터 타입은 category 
    category 는 현재 상태에서 고정된 카테고리가 3개가 있으면 수정이 안됨 (뭔소린지는 잘 모르겠음)  
    따라서 타입을 변경해준다 **astype()**
  - df1.deck = df1.deck.astype('object')
  ![image](https://user-images.githubusercontent.com/82145878/178248088-5eec491c-7c34-4321-94cf-d6d1957c2fd2.png)  
  ==> __정상작동!__  

  
* df.corr()
 - 상관관계 : 얼마나 y값을 x값이 잘 설명하고 있느냐 min = 0 max = 절대값 1  
 - 양의 상관관계 x증가 y __증가__  
 - 음의 상관관게 x증가 y __감소__  
      => 결국 둘다 관계성이 있고 방향만 다른것임  
 - 0에 가까우면 y에 있어서 x값이 잘 설명하고 못하고 있다는 것 : __관계성이 낮다__  
 
 
 * one-hot-encoding
 - 회귀 분석을 할 때 `원핫인코딩`은 필수
 - Label encoding 의 문제 : unique 값이 1000개라면 .. 마지막 행의 값은 1000이 된다. 그러나 회귀분석은 숫자의 크기에 영향이 굉장히 많이 받음. 
 - 숫자가 커질수록 가중치가 줄어들음. 
 - 인코딩에서 1과 1000의 숫자 차이는 아무 의미 없지만 회귀분석에서는 문제가 생겨 분석이 안될 수 있음
 - 사이즈를 똑같이 맞추려면 어떻게 해야할까? ==>  원핫인코딩
 - 즉 전체사이즈가 1로 맞춰짐 분류는 원핫인코딩을 안해도 영향을 안받아서 ㄱㅊ음 그러나 회귀분석은 안된당 ㅋㅋ.
 
* preprocessing 의 labelencoder라는 클래스
- fit == > 정렬 sorting (오름차순) 
- transform 첫번째 데이터부터 순서대로 0번부터 할당을 해줌
- inverse_transform을 하면 디코딩... value값으로 key 값을 찾음  (dictionary 라서)
```python
from sklean.preprocessing import LabelEncoder
encoder = LabelEncoder()
#데이터 가져오기
items = ['tv','냉장고','전자레인지','컴퓨터','선풍기','선풍기','믹서기','믹서기'] #items는 컬럼명
encoder.fit(items)
#정렬하고..중복없애고.. encoder에 저장되어있음
encoder.transform(items)
labels = encoder.transform(items)
```
![image](https://user-images.githubusercontent.com/82145878/178211674-49c8e08a-e1ba-44b9-9a83-765eadaaa1cf.png)  
    ==> labels 출력
    
    






  
