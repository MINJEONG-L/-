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
  
  
# 2일차  
  
## Feature Scaling  

1) 표준화 : STANDARDSCALER
  ![image](https://user-images.githubusercontent.com/82145878/178380580-c84e1145-0bb1-40cb-a88a-c72190034942.png)  

<방법1>
```python
def standard(x):
  return (x-x.mean())/(x.std())
```  

<방법2>
```python
from sklean.preprocessing import StandardScaler
#StandardScaler 객체 생성
scaler = StandardScaler()

#StandardScaler로 데이터 세트 변환. fit()과 transform()호출
scaler.fit(df_train)
first_scaled = scaler.transform(df_train)

#transform() 시 스케일 변환된 데이터 세트가 Numpy ndarray로 반환돼 이를 DataFrame으로 변환
first_df_scaled = pd.DataFrame(data=first_scaled, columns = list(df_train.columns))
print('feature들의 평균값')
print(first_df_scaled.mean())
print('\nfeature들의 분산값')
print(first_df_scaled.var())
```  
![image](https://user-images.githubusercontent.com/82145878/178484674-40fb7ee3-f7ce-4707-ad4e-35e5f409fa6a.png)  


2) 정규화  
  ![image](https://user-images.githubusercontent.com/82145878/178380570-aaf63d90-4e49-4994-ab15-2cc67e94fb99.png)  
  

>> 머신러닝 전에 상관관계를 다시보기
>> 문자데이터가 있어서 상관관계를 볼 수 없엇는데 지금은 인코딩햇으니까 볼 수 있음  


## Machine Learning - 분류(Classification)  

  * 데이터 분리 : 학습데이터(train) + 테스트데이터(test)  
  ```
  ex) X_train.shape   => (712,7)
      y_train.shape   => (179,7)
      X_test.shape    => (712,)
      y_test.shape    => (179,)
  ```  
  
  * Scikit Learn 사이킷런 - 모델 선택
    - `Random Forest Simplified`,  `Logistic Regression`,  `Decision Tree Classifier`  
    
    ![image](https://user-images.githubusercontent.com/82145878/178486110-078e30e7-37e6-481f-ba92-61111edb45d2.png)  
    
    ```python
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    dt_clf = DecisionTreeClassifier(random_state = 11)
    rf_clf = RandomForestClassifier(random_state = 11)
    lr_clf = LogisticRegression()
    ```  
    
  * 교차 검증 KFold
    ![image](https://user-images.githubusercontent.com/82145878/178488976-58883336-9f0c-463e-bfc7-4310d8872297.png)  

   ```python
   from sklearn.model_selection import KFold
   def exec_kfold(clf, folds=5):
   #폴드 세트를 5개인 KFold 객체를 생성, 폴드 수 만큼 예측결과 저장을 위한 리스트 객체 생성
   kfold = KFold(n_splits = folds)
   scores = []  
   #KFold 교차 검증 수행
   for iter_count, (train_index, test_index) in enumerate(kfold.split(X_titanic_df)):
   #X_titanic_df 데이터에서 교차 검증별로 학습과 검증 데이터를 가리키는 index 생성
   X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
   y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
   #Classifier 학습, 예측, 정확도 계산
   clf.fit(X_train, y_train)
   predictinos = clf.predict(X_test)
   accuracy = accuracy_score(y_test, predictions)
   scores.append(accuracy)
   print("교차 검증 {0} 정확도 : {1:4f}".format(iter_count, accuracy))
   
    #5개 fold에서의 평균 정확도 계산
    mean_score = np.mean(scores)
    print('평균 정확도:{0:.4f}".format(mean_score))
   ```  
  
    * 만약 1의 분포가 5% 내외라서 일부 검증 레이블 데이터 분포가 극단적이라면?
      - 예) 0:880 , 1:11 개 라면 아주 극단적으로 한 데이터셋에 음성이 전부 들어갈 수도 있음
      - positive, negative ratio 를 계속 가지고 가자! ==> stratifiedKFold
   
        `kfold는 x나 y나 아무거나 넣어도 됨` : 둘 다 같은 인덱스를 가지고 있어서
         **그러나** `skfold에서는 y값이 꼭 필요`  
         
  * 교차 검증 StratifiedKFold  
  
  ```python
  from sklearn.model_selection import StratifiedKFold
  
  skf = StratifiedKFold(n_splits = 5)
  num = 0
  
  for train_index, test_index in skf.split(df_train, df_train['Survived']):
    num += 1
    label_train = df_train['Survived'].iloc[train_index]
    label_test = df_train['Survived'].iloc[test_index]
    print("교차 검증: {0}".format(num))
    print('학습 레이블 데이터 분포:\n', label_train.value_count())
    print('검증 레이블 데이터 분포:\n', label_test.value_counts()) 
  ```  
  
  ```python
  skfold = StratifiedKFold(n_splits = 5)
  n_iter = 0 
  cv_accuracy = []
  
  #StratifiedKFold의 split() 호출 시 반드시 레이블 데이터 세트도 추가 입력 필요
  for train_index, test_index in skfold.split(X_titanic_df, y_titanic_df):
    #split() 으로 반환된 인덱스를 이용해 학습용, 검증용 테스트 데이터 추출
    X_train, X_test = X_titanic_df.values[train_index], X_titanic_df.values[test_index]
    y_train, y_test = y_titanic_df.values[train_index], y_titanic_df.values[test_index]
    #학습 및 에측
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    
    #반복 시마다 정확도 측정
    n_iter += 1
    accuracy = np.round(accuracy_score(y_test,pred),4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('\n#{0} 교차 검증 정확도 :{1}, 학습 데이터 크기: {2}, 검증 데이터 크기: {3}'format(n_iter, accuracy, train_size, test_size))
     cv_accuracy.append(accuracy)
    # 교차 검증별 정확도 및 평균 정확도 계산
     print('\n## 교차 검증별 정확도:', np.round(cv_accuracy, 4))
     print('## 평균 검증 정확도:', np.mean(cv_accuracy))     
  ```  
  
  * 교차 검증 Cross_Val_Score
  ```python
  from sklearn.model_selection import cross_val_score
  
  for MLMD in [dt_clf, rf_clf, lr_clf]:
    scores = cross_val_score(MLMD, X_titanic_df, y_titanic_df, cv = 5)
    for iter_count, accuracy in enumerate(scores):
      print("교차 검증 {0} 정확도: {1:.4f}".format(iter_count, accuracy))
    print("평균 정확도: {0:.4f}".format(np.mean(scores))
  ```

  >> kfold가 남아있는 이유 : 근본적으로 skfold는 positive, negative ratio 비율을 확인하기 위해서 y값이 꼭 필요함
  >> skfold는 연속값을 측정할 때 (회귀)는 사용할 수 없음 ==> 모든 비율을 측정할 수 없기 때문에
  >> 회귀에서는 kfold를 씁니다 이진분류면 skfold 가능  


 
## 최적 하이퍼 파라미터

   - 하이퍼 파라미터를 손대서 교차검증을 하겠다!  `하이퍼 파라미터 튜닝`은 필수!!  
   
   ```python
   from sklearn.model_selection import GridSearchCV
   parameters1 = {'max_depth':[1,2,3,4,5,6,7,8,9,10],
                  'min_samples_split':[2,3,4,5,6,7,8,9,10], 'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]}
   grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv = 5)
   grid_dclf.fit(X_train, y_train)
   
   print("GridSearchCV 최적 하이퍼 파라미터:', grid_dclf.best_params_)
   print("GridSearchCV 최고 정확도 : {0:.4f}'.format(grid_dclf.best_score_))
   best_dclf = grid_dclf.best_estimator_
   
   #GridSEarchCV 의 최적 하이퍼 파라미터로 학습된 Estimator로 예측 및 평가 수행
   dpredictions = best_dclf.predict(X_test)
   accuracy = accuracy_score(y_test, dpredictions)
   print('테스트 세트에서의 DecisionTreeClassifier 정확도 : {0:.4f}'.format(accuracy))
   ```  
   
   ![image](https://user-images.githubusercontent.com/82145878/178518484-b9c22406-f08d-4679-9ca6-5ae4ec0df9c4.png)  
   
   
# 3일차  

## wine data 활용하기  
  - 드라이브에서 데이터 불러오기
  ```python
  df_red = pd.read_table(path + 'winequality-red (1).csv' ,sep=';') #table csv 둘다 맞음
  df_white = pd.read_csv(path + 'winequality-white (1).csv', sep=';' )
  ```  
  
cv summary.....뭐지..

## 평가 및 적용 : 이제 머신러닝 평가를 해보자   

  * 평가지표  

    `네가지의 녹색 부분만 신경을 쓰자`  

    ![image](https://user-images.githubusercontent.com/82145878/178638967-7718c539-420b-4340-baa1-c4a021630591.png)  

    - true posivite(TP) -> positive 로 예측했는데 true 였다.  
    - true negative(TN) -> negative 로 예측했는데 true 였다.  
    - false posivite(FP) -> positive 로 예측했는데 false 였다.  
    - false negative(FN) -> negative 로 예측했는데 false 였다.  
    
    
   1) 회귀 성능 평가 지표  
    : MAE, MSE, RMSE, R^2..  
    
   2) 분류 성능 평가 지표  
    : 정확도, 오차행렬, 정밀도, F1스코어, ROC AUC... 
    
      (1) 정확도 Accuracy  
    
        - (예측 결과과 실제 결과와 동일한 데이터 건수) / (전체 예측 데이터 건수)  
    
       ```python
       from sklearn.base import BaseEstimator
       ```  
    
      (2) 오차행렬 Confusion Matrix  
    
        ```python
        from sklean.metrics import confusion_matrix
        confusion_matrix(y_test, mypredictions) #confusion_matrix(실제값, 예측값) 
        ```  
    
        - 출력값이 초록색 네모(TP, TN, FP, FN)에 맞춰서 나옴
        - FP 는 잘 안나옴, TP는 거의 의미가 없다(정상인을 정상인이라고 맞추는 것은 의미가 없음)  
        - FN 은 임산부인데 임산부가 아니라고 한것  
            ==> 제일 위험! FN을 낮추는게 젤 중요  
        - 찾고자하는 것 : `positive`  
        
        
        
        
        ||PRE|PRE|PRE|
        |---|---|---|---|
        |ACT||0|1|
        |ACT|0|TN|FP|
        |ACT|1|FN|TP|
        
        
        
      (3) 정밀도(Precision)& 재현율(Recall)  
    
        - 정밀도, 재현율에서 TN은 거의 사용 x  
        - 정밀도 : `예측값`이 Positive 인 것 중에서 TP를 보는 것  **FP가 중요**  
                실제 음성인 데이터 예측을 양성으로 잘못 판단 시 업무상 큰 영향이 발생하는 경우  
              ![image](https://user-images.githubusercontent.com/82145878/178678230-efd67447-971b-406b-89c2-c6aa9e147f50.png)  

        - 재현율 : `실제값`이 Positive 인 것 중에서 TP를 보는 것  **FN이 중요**  
              실제 양성 데이터를 음성으로 잘못 판단 시 업무 상 큰 영향이 발생하는 경우  
              ![image](https://user-images.githubusercontent.com/82145878/178678287-e4e858b8-b393-4224-b91b-c295d103f23d.png)  
         
        - FP 커질수록 정밀도가 낮아지고 FN이 커질수록 재현율이 낮아진다. 즉 반비례한 관계로, 그 `절충`을 찾는 것이 중요  
          ex) 암판단은 `재현율`이 중요!  
          ex) 법률 판단은 `정밀도`가 중요
        `confusion_matrix` **만 알면**  `precision_score`, `accuracy_score`, `recall_score`, **쉽게 구할수 있음**  
      
        ```python
        from sklean.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
        def get_clf_eval(y_test, pred):
          confusion = confusion_matrix(y_test, pred)
          accuracy = accuracy_score(y_test, pred)
          precision = precision_score(y_test, pred)
          recall = recall_score(y_test, pred)
          print('오차 행렬')
          print(confusion)
          print('정확도: {0:.4f}, 정밀도: {1:.4f}, 재현율:{2:.4f}'.format(accuracy, precision, recall))
        ```  
        
       [-] TRADE-OFF  
            ![image](https://user-images.githubusercontent.com/82145878/178679362-21f5edc7-7a5b-49f0-b1de-d28a6c827d6f.png)  
          - threshold 값보다 작으면 0 크면 1 이런 식..  
            
         ```python
         from sklearn.preprocessing import Binarizer
         X = [[1,-1,2],
             [2, 0, 0], 
             [0, 1.1, 1.2]]
         binarizer = Binarizer(threshold = 1.1)
         print(binarizer.fit_transform(X))
         ```  
                
      ![image](https://user-images.githubusercontent.com/82145878/178715031-a766dbaa-c25d-4505-b60c-a7b2cc0dc28b.png)  
      ![image](https://user-images.githubusercontent.com/82145878/178715104-7f2d0afd-3974-4e36-aa30-974cee3a6c4c.png)  
    
       ```python
       thresholds = [0.4,0.45,0.5,0.55,0.6]
         
       acc = []
       pre = []
       re = []
       f1 = []
       def get_eval_by_threshold(y_test, pred_proba1, thresholds):
          for custom_threshold in thresholds:
             binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba1)
             custom_predict = binarizer.transform(pred_proba1)
             a,b,c,d = get_clf_eval(y_test, custom_predict)
             acc.append(a)
             pre.append(b)
             re.append(c)
             f1.append(d)
             get_eval_by_threshold(y_test, pred_proba[:,1].reshape(-1,1), thresholds)
       ```  
     
      ![image](https://user-images.githubusercontent.com/82145878/178680147-e6ad8089-f054-484e-8789-ea32696a0b96.png)  
      ![image](https://user-images.githubusercontent.com/82145878/178680354-a3f78290-c00e-4484-9335-86bea7ef30fb.png)  
      
      (4) F1 스코어
       - f값  
       ![image](https://user-images.githubusercontent.com/82145878/178716191-0acf4128-7e41-45f1-8bbb-8400bd54b31d.png)  
       
       
       ||precision|recall|F1|
       |---|---|---|---|
       |A|0.9|0.1|0.18|
       |B|0.5|0.5|0.5|
       
       
        ==> 둘 중 하나만 크면 안되고 둘 다 적당해야 f값이 작아지지 않음  
        
       (5) ROC AUC  
       - ROC : Receiver Operation Characteristic Curve 수신자 판단 곡선  
       - 머신러닝 이진 분류 모델의 예측 성능 판단에 중요한 평가 지표
       ![image](https://user-images.githubusercontent.com/82145878/178719296-e2aefe82-dda8-4897-9824-f1c5aff9e529.png)  
       - TPR = TP/(FN+TP) 재현율  
       - TNR = TN/(FP+TN)
       - FPR = FP/(FP+TN)
       - FPR = 1 - TNR 
       >> FPR -> 0  
          - 분류 결정 임곗값 -> 1  
          - FP/(FP+TN)  
          - Positive 예측 기준이 Max  
          - Positive가 틀릴 확률 0  
          - Positive 선택 0  
          - FP는 항상 0  
          
          
       >> FPR -> 1  
          - FP/(FP+TN)
          - 분류 결정 임계값 -> 0  
          - 모두 Positive로 예측  
          - Negative 예측 확률이 0 -> TN = 0  
          
       - AUC : Area Under Curve  
       ![image](https://user-images.githubusercontent.com/82145878/178721883-b0f8a459-9655-4a1b-a816-7f99766454ed.png)  

       - ROC 곡선 밑의 면적  
       - 1에 가까울 수록 좋은 수치  
       - FPR이 작은 상태에서 TPR이 커지는 것  
       - 가운데 대각선은 동전 던지기 수준의 AUC 값  
       - 보통 분류는 0.5이상의 AUC 값  
       
          ```python
          from sklearn.metrics import roc_auc_score
          pred = lr_clf.predict(X_test)
          roc_score = roc_auc_score(y_test, pred)
          print("ROC AUC 값 : {0:.4f}'.format(roc_score))
          ```  
          

       ![image](https://user-images.githubusercontent.com/82145878/178671990-f9e93768-14dc-4215-897e-74fcaa19eff9.png)  
       ![image](https://user-images.githubusercontent.com/82145878/178672034-f0b89c21-ebe5-40a9-8062-de4311bf7053.png)  
       ![image](https://user-images.githubusercontent.com/82145878/178671961-c04789fb-0419-4519-8c81-4071e36d4a72.png)  
    
  
  
      



<img src=https://user-images.githubusercontent.com/82145878/172035324-2b35272c-1325-4a77-9016-bbd2d3048be9.png width="80%" height="80%"/>  
  
