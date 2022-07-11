# 머신러닝

## 빅데이터 분석절차
```mermaid
graph LR
A(기획:목적)-->B(필요한 데이터 수집 Data Collection)
B-->C(데이터 전처리 Data Preprocessing)   
C-->D(모델 선택 Model Selection)  
D-->E(평가 및 적용 Evaluation & Application) 
```  


## titanic data 실습  
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
    >> **groupby**  
   
    df.groupby 특정조건을 그룹으로 묶어서 특정조건에 따라 여러 개의 데이터 프레임으로 쪼개지고 (출력은 안됨) 수정가능  
      
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
    ![image](https://user-images.githubusercontent.com/82145878/178252330-53d62e66-0494-42c7-a044-4629bc00fb0e.png)  
    * heatmap으로 표현하기  
    
    ```python
    sns.heatmap(df.corr(), cmap=None, annot = True, cbar=True)
    # 데이터를 쓰고 싶다 annot = True
    ```  
    ![image](https://user-images.githubusercontent.com/82145878/178252624-674793c1-b2c1-4f0d-9d25-c66d81b06af0.png)  
    ```python
    sns.heatmap(df.corr(), cmap='coolwarm', annot = True, cbar=True)
    # 데이터를 쓰고 싶다 annot = True
    ```  
    ![image](https://user-images.githubusercontent.com/82145878/178252721-3ebff9dd-1fd1-4bc2-9257-c6c84398c338.png)  
    
* crosstab  
  - 데이터 분석을 하다가 원본 데이터의 구조가 분석 기법에 맞지 않아 행, 열의 위치를 바꾸거나 특정 요인에
    따라 집계를 해서 구조를 바꿔주어야 하는 경우  
     **재구조화(reshaping data)**  
  - 재구조화 함수
    `pivot(), pd.pivot_table()`, `stack(), unstack()`, `melt()`, `wide_to_long()`, `pd.crosstab()`  
    
    (1) 교차표 만들기 : `pd.crosstab(index, columns)`  **행과 열 위치에는 array 형식의 데이터**  
    (2) Multi-index, Multi-level로 교차표 만들기
          `pd.crosstab([id1,id2],[col1,col2])`  
    (3) 교차표의 행 이름, 열 이름 부여 : `pd.crosstab(rownames=['xx'], colnames=['aa'])`  
    (4) 교차표의 행 합, 열 합 추가하기 : `pd.crosstab(data.id, [data.fac_1, data.fac_2], margins=True)`  
    (5) 구성비율로 교차표 만들기 : `pd.crosstab(data.id, [data.fac_1, data.fac_2], normalize=True)`  
    
    ![image](https://user-images.githubusercontent.com/82145878/178279498-dd2502d0-78a0-4f03-8e7b-67c097272958.png)  
    
    >> violinplot(x축, y축, hue=결과값, data=데이터)  
    
      ![image](https://user-images.githubusercontent.com/82145878/178280198-8ad6b2be-8a08-451b-a730-a50d175081e5.png)  
      
    >> 하나로 합치기 split=False를 True  
   
      `sns.violinplot(x='embarked', y='age', hue= 'survived', data=df, split = True)`  
      
      ![image](https://user-images.githubusercontent.com/82145878/178280413-c6bd85db-8fc7-4cdd-b784-360f7ab50f6e.png)  
      
    >> inner = 'quartile' 이면 MIN MAX MED 값 표시  
   
      `ns.violinplot(x='embarked', y='age', hue= 'survived', data=df, split = True, inner = 'quartile')`  
      
      ![image](https://user-images.githubusercontent.com/82145878/178280605-383a8359-249b-4d96-86e1-f8e23a8690d9.png)  
     
   **결측치 제거하고 이상치 제거도 좋긴 하지만 이상치를 제거하지 않아도 머신러닝 돌릴 수 있음
      encoding 안하면 머신러닝 자체가 안돌아감. 즉 문자를 숫자로 바꾸는 작업 ENCODING**  
      
   **회귀 분석에서는 원핫인코딩이 필수임 분류는 아니지만.**  
      - label 인코딩에서 끝나는게 아니라 원핫인코딩을 하는 이유는?  

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
  `encoder.classes_  #중복제거 오름차순 `  
  
  `encoder.inverse_transform([2]) #매개변수가 1차원 array 여야함`  
  
  >> get_dummies() 함수  

  get_dummies() 함수를 사용하면 바로 원핫인코딩 가능  

  `pd.get_dummies(items)`  
  
  ![image](https://user-images.githubusercontent.com/82145878/178283639-6ab96ab1-ce39-41f9-b4ec-83c26f8bc58f.png)  
  
* df의 숫자가 아닌 데이터 타입들을 라벨링  

  ```python
  enlist = df.dtypes[(df.dtypes == 'object') | (df.dtypes == 'bool') | (df.dtypes == 'category')].index
  for i in enlist:
    encoder = LabelEncoder()
    encoder.fit(df[i])
    df[i] = encoder.transform(df[i])
  ```  
  ![image](https://user-images.githubusercontent.com/82145878/178285226-7a67fcd9-5c21-4a0d-9e39-bd34fc40ac33.png)  
  
  ![image](https://user-images.githubusercontent.com/82145878/178285408-034cad02-8286-44b5-960d-c3d4948d0e83.png)  
  








  
