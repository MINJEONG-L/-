# Regression  
![image](https://user-images.githubusercontent.com/82145878/179225855-e6241307-563f-4e51-a5ee-e335bbf2e8aa.png)  
`y = w1x1`  
- 회귀의 목적 : 추세선, 최적의 w를 찾는 것, 가장 설명을 잘하는 직선을 찾자  
![image](https://user-images.githubusercontent.com/82145878/179226974-f1d56940-62c0-4d30-a416-b1196b55cd26.png)  

- 집단은 평균값으로 다시 돌아오게 되어있다.  
- 에러는 실제값-예측값  
 ![image](https://user-images.githubusercontent.com/82145878/179227098-beca05c1-8766-4514-bb2b-a777a6cf8fa4.png)  

- 통상적으로 에러의 제곱을 더 많이 쓴다 SE(Square Error) , MSE(Mean Square Error) 에러제곱의 합의 평균  
- 에러가 커지니까 (제곱하면) 조금만 줄여도 에러를 확확 줄일 수 있게 방향 설정 가능  
- `Mean Square Error` == `Cost function 비용 함수(코스트는 줄여야함)` == `RSS(w) 잔차제곱합` 
- Acc 스코어는 100을 향해 하고 **MSE는 0**을 향해가고  
- Cost function 의 `최소값`이 되는 하나의 w를 찾는게 목적  
- 제곱의 합이니까 제곱의 합으로 올라가게 됨.  어쨋든 RSS는 양쪽으로 `제곱의 합`만큼 올라감  
- `미분이 0`인 지점을 찾아가야함  
- 얼마만큼 가야 0까지 갈 수 있는지 모름 가파른지 완만한지 모름 예) 0.9에서 얼마나 줄여야하는지 잘 모르는것  
- 미분을 했더니 +값이 나오거나 -값이 나오거나 하면 어디로 줄어야할지 방향은 알음  
- **-**가 나오면 w를 **늘려야한다** **+** 가 나오면 w를 **줄여야한다.**  

-  로지스틱에 learning rate가 있었잖아 이거 가중치를 얼마나 올리고 내릴지를 말하는 거엿음  
-  그래서 너무 작으면 오랜시간이 걸린다  
-  학습이 아예안되면 learning rate를 낮춰서 속도를 느리게 하고..  
        ==> 이런건 `하이퍼파라미터`임  
 **경사상승이 되서는 안된다**  
-  어떻게 이걸 자동으로 검색하게 할가 
![image](https://user-images.githubusercontent.com/82145878/179228065-c87bfb07-a1d1-477e-bb44-5b72a5aeb32c.png)  

- RSS(w1)을 미분하면 1/N 시그마(-2yixi + 2wx(xi)^2) == (-2/N 시그마 (yi-xi-w(xi)^2))  
- ^y 는 예측값임  
- 이 식을 자동화를 시켜야함  
- 어쩃든 앞에 마이너스까지 포함한 결과가 +면 줄어야하고(빼야하고) -면 더해야함.  
![image](https://user-images.githubusercontent.com/82145878/179228712-b96ee634-35d6-42f9-82bc-541a41ef5904.png)  

- σr(w)/σw를 왜 뺴냐면 기울기가 작았다 == w를 찾는 과정  
- 점점더 작은 값을 빼주는 것임  
- w를 찾는 기준이 `경사하강법` 임     *사실 1차회귀할때는 경사하강법 안써도된다  연립방정식 써도 w를 구할 수 있더..*    
- bias는 절편..?  
- 변수가 두개이상인데 미분을 어케 하나요> ==> `편미분`   
![image](https://user-images.githubusercontent.com/82145878/179228878-8abc8473-76a3-4dd6-89cd-43ab82907beb.png)  

- RSS(W0,W1) = 1/N 시그마(아이는 일부터 엔까지) (yi-^yi)^2  = 1/ㅜ  
![image](https://user-images.githubusercontent.com/82145878/179228971-041856d4-0bd9-4df5-8ff5-56417fc6be57.png)  
![image](https://user-images.githubusercontent.com/82145878/179229069-6d43655c-4c22-411b-8ea2-3fb8ad258180.png)  
![image](https://user-images.githubusercontent.com/82145878/179229116-8f1bbec1-8887-478f-9747-9f74496fe4c2.png)  

-  앞시점의 w1에서 w1미분한 값을 빼준다 
-  양수 음수 알아서 ㅇㅇ  
-  계산량이 많은거지 계산이 복잡한것은 아님   
-  w0 w1 둘다 동시에 값이 업데이트  
![image](https://user-images.githubusercontent.com/82145878/179229234-feab4a6f-5959-4e15-8eec-6140b98057ac.png)  

-  약속한 값에 도달하면 그 시점의 w0 w1 값을 보면 최적의 w를 구한것이다  `절편도 기울기`도 바뀐다.  
 ![image](https://user-images.githubusercontent.com/82145878/179229402-95269f8a-4d95-4f08-b848-f064ccca172b.png)  

 ![image](https://user-images.githubusercontent.com/82145878/179229484-f9a4c648-8cc2-445e-b6a7-57e9c4f80378.png)  
 ![image](https://user-images.githubusercontent.com/82145878/179229592-d86c9c61-e334-40b3-aff1-eee6c010b9d4.png)  
  
- 출력값은 연속값  
- 변수(피처)가 n개면 종속변수 1개 빼고 나머지 xtrain에 들어가는 변수를 넣으면 12개의 x값이 잇고 그거에 대응하는 w값이 잇는데
- 뭐가 하나 더 생겨서 13개 편미분을 함  
![image](https://user-images.githubusercontent.com/82145878/179229669-5ec3780a-fca7-4dec-bf54-7e3c45fd2dd4.png)  

# Sklearn 사이킷런   
`sklearn.linear_model.LinearRegression(Parameter)`  
  *회귀는 사이킷런에 아주쉽게 구현 !!*  
  linear regression 선형 회귀  
  - fit_intercept : True/False 는 디폴트가 True, 절편 값을 계산할 것인지 여부, False : 절편이 0으로 지정  
  - 당연히 true 절편은 무조건 계산 해야짐  
  - normalize:True/False는 디폴트가 False  
  - 굳이 feature 스케일링을 안해도 normalize = True 해주면 회귀를 수행하기 전 입력데이터 세트를 정규화  
* 회귀평가지표  
1) MAE (Mean Absolute Error)
2) MSE (Mean Squared Error)  
3) RMSE(Root Mean Squared Error)  
4) R^2  
* 주로 R^2 MSE RMSE 누가 더 0에 에러가 가깝게 예측을 햇느냐가 평가 지표가 된다.  
* 회귀 평가 지표 MAE는 잘 안씀  




