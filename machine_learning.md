# machine learning

## Ensemble  
  - 단순/가중 평균(Simple/weighted average)  
  - 배깅 (Bagging = Bootstrap aggregating)  
  - 부스팅 (Boosting)  
  - 스택킹 (Stacking)
  - 메타학습 (Meta-learning)  
  
  (1) 단순 가중 평균 VOTING  
  ![image](https://user-images.githubusercontent.com/82145878/178984388-476c4a66-1064-4d2a-ac70-68debdf4a614.png)  
  (2) 배깅 Bagging  
  
  (3) 부스팅 Boosting  
  - 부스팅의 대표적인 구현은 AdaBoost(Adaptive boosting 적응형)와 GBM(Gradient Boost Machine)이 있다.  
  - 모델 자체가 직렬이라 오래걸림.  
    **AdaBoost(Adaptive boosting 적응형)**
     ![image](https://user-images.githubusercontent.com/82145878/179017828-109e7394-96d1-4376-99ef-a90127bea4f7.png)  
     - +,-로 피처 데이터셋이 존재하고 +를 탐지하려 했으나 탐지하지 못하게 되면 다음 순서에서와 같이 +의 크기를 키워(가중치를 키워) 다음 약한 학습기가 더 잘 분류할 수 있도록 진행  
    **GBM(Gradient Boost Machine)**  
    - AdaBoost와 유사하지만 가중치 업데이트를 경사 하강법을 이용하는 것이 큰 차이
    - 하이퍼파라미터 튜닝 : `Loss`, `learning_rate`, `n_estimators`, `subsample`  
    - 장점 : 과적합에도 강하고 뛰어난 예측 성능  
    - 단점 : 수행시간이 오래 걸림.  
  - 지금까지 왜 Boosting이 Bagging 에 밀렸는가?
      => 너무 많은 계산량, 튜닝해야 할 파라미터가 너무 많다.
          bagging은 분산 컴퓨팅이 가능하지만 boosting은 분산 컴퓨팅이 어렵다.  
          
     **XGBoost(eXtreme Gradient Boost)**  
     - GBM에 비해 추가된 파라미터들  
        : 뛰어난 에측 성능, GBM 대비 빠른 수행 시간, 과적합 규제(Regularization), Tree pruning, 자체 내장된 교차 검증, 결측값 자체 처리  
     - 특이사항 : 조기중단 기능 early_stoppings  
     
     **LightGBM**  
     - Leaf Wise  
      (1) XGBOOST Level-wise tree growth  
      ![image](https://user-images.githubusercontent.com/82145878/179021068-c06ca22a-41b7-42b1-8fa9-c45b6e73aa34.png)  
      - 데이터가 적어도 돌릴 수 있음
      (2) LightGBM Leaf-wise tree growth  
      ![image](https://user-images.githubusercontent.com/82145878/179021174-64507a78-8e2f-41db-a92f-5f255c91343e.png)  
      - 한쪽 노드만 잡고 계속 내려간다.  ==> 속도가 빠름  
      - 데이터가 적으면 어떤 것만 잘 맞춰서 모델이 완성이 안된다. 어쩔땐 잘 나왔다가 안나왔다가 그럼  
      - 
      
## Undersampling & Oversampling  

![image](https://user-images.githubusercontent.com/82145878/179022527-83f644db-4c07-44cc-a667-02dd9117d753.png)  
  1) 언더 샘플링 (under sampling)  
    - 많은 데이터 -> 적은 데이터 수준  
    - 예) 0:10,000건, 1:100건 -> 0:100건, 1:100건  
    - 과도하게 정상 레이블로 학습/예측하는 부작용 개선  
    - 정상 레이블의 학습을 제대로 수행하기 어려운 단점  
   
  2) 오버 샘플링 (over sampling)  
  ![image](https://user-images.githubusercontent.com/82145878/179022720-4e7142e8-4967-4fc5-a816-c2565b2beb63.png) 
    * SMOTE PROCESS  
     - 대표적인 오버 샘플링 기법 중 하나이다.  
     ![image](https://user-images.githubusercontent.com/82145878/179023195-a8d12f58-de01-405c-a2cd-5d8dca36d486.png)  
     - 수가 적은 positive 데이터의 개수를 늘리는 것  
     - negative쪽이 아니라 positive 데이터들 사이의 선 위에...  
     - 그런데 어차피 5개의 바운더리 안에 들어가는 거기 때문에 안만들어도 영역을 찾고자 하는 것이지 데이터를 늘리고자 하는 것은 아니다.  
     - 이 안에 들어가는 것을 찾고싶은 것은 아닌 것임. 데이터를 늘린 다는 것이 주목적  
     - 영역 구분은 확실하게 될 것임 그러나 아예 다른 곳에 찍혔던 것을 찾고 싶었던 것!  
     ==> 그래서 결론이 뭐냐  
     
## 스태킹 앙상블 Stacking Ensemble  
  ![image](https://user-images.githubusercontent.com/82145878/179024951-09ee72b7-aaaa-4e93-a9d6-3c339c51d7df.png)  
  - stacking 쌓다  
  - 사진의 4개 모델 모두 잘 만들어진 모델들  
  - m개의 데이터가 들어오면 각각의 모델들은 m번을 예측을 하겠지 : m번 학습  
  - M개는 모델의 수  
  - 데이터의 개수 * 모델의 수 만큼 데이터 프레임이 만들어짐  
  - 새로운 데이터 프레임이 생기면? ==> 학습할 수 있음  
  - 왜 또 학습?  
    - 0000으로 들어오면 0이라고 예측하는게 맞음  
    - 근데 0001로 들어오면 0이라고 예측 그런데 만약 답이 1이라면?  
    - voting은 무조건 틀림 다수결이기 때문에  
    - 그러나 스태킹은 이걸 학습한다 소수의 의견이라도 맞추면 M4에게 전문성을 주겠다(가중치?)  
    - 전문성을 어떻게 확인하느냐? 정답(y_test)과 비교  
    - 
  

    
  


        
