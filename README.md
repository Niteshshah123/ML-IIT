# ML-IIT

# ML types
 - ##### SuperVised Learning (Input and outpus is Known, just we have to train the model based on in/output we have).
    - 1. ###### Regression Learning (Used for Contineous dataSet)
          - 1.1 ``linear Regression(y = mx+c)``
                - 1.1.1 Multiple Linear
                - 1.1.2 Polynomial Regression (degree >= 2)
         - 1.2 ``Decission Tree`` (Problem of OverFitting) ![Youtube](https://youtu.be/CWzpomtLqqs?si=bwEOwsJDv7I67kK5) 
         - 1.3 `` Random Forest (Collection of  Decision Tree)`` (Overcome the problem of overfitting (Decision Tree)) (Mostly Used in Classification)<br>
           `` **Ans will be Majority among different dicision Tree!** ``
           ![image](https://github.com/user-attachments/assets/1984413e-70f7-4efd-a5ed-c6ba07189d8f)
         - 1.4 `` Neural Network ``
    - 2. ###### Classification Learning (Divide the output into 2 set) (used for Discrete dataset)
          - 2.1 Logistic Regression (For Binary output)
          - 2.2 SVM ()
            ![image](https://github.com/user-attachments/assets/c9b56fa9-1f04-4589-b1c1-fb0e1025c45a)
          - 2.3 Naive Bayes ![Youtube](https://youtu.be/GBMMtXRiQX0?si=eO202-7R-5wzmWx3)
            ![image](https://github.com/user-attachments/assets/2252e8a3-0a43-4b2d-a075-801deafa86f7)
            ![image](https://github.com/user-attachments/assets/2764ae3a-cfd2-4221-8f84-476892d76b1b)
          - 2.4 ``Decision Tree``, ``Random Forest``, ``Neural Network`` 

 - ##### UnSuperVised Learning (No Output, Search Pattern in Input and group them based on Similar Pattern)
   - Clustering
     ![image](https://github.com/user-attachments/assets/87aa9e35-b28d-40c1-8fb7-8fe882599550)
   - Dimensionality Reduction
     ![image](https://github.com/user-attachments/assets/a03e64c0-f29e-46bd-b68e-1623f7949c22)

--- 
## Project

### Sonar Rock Prediction

#### when to use fit only and when to use both fit_transpose
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)          # Learns mean and std of the features
X_train_scaled = scaler.transform(X_train)  # Applies scaling
X_test_scaled = scaler.transform(X_test)    # Uses same scaling rules
```

``x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)`` here stratify=y means that, the ration of rock and mine will be same in both test and split.

