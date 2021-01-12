# Homework of Machine Learning 109F
> Lectured by Prof. Jen-Tzung Chien
> and this note is owned by Wen-Sheng Lo
### HW1
1. Bayesian Linear Regression
    - Prove the predictive distribution of Bayesian linear regression
    - See "Pattern Recognition and Machine Learning" of Christopher M. Bishop  for more
2. Linear Regression
    - Need to know the difference of MLE(Maximum Likelihood Estimation) & MAP(Maximum A Posterior)   
    - See [this article](https://www.ycc.idv.tw/deep-dl_3.html) to learn more
    - MLE
        - No need too much assumption
        - Maximize the likelihood (Minimize the KL divergence)
        - Try to find the model's distribution from data
        - Might have overfitting problem
        - $\theta_{MLE} = argmin_{\theta}\frac{1}{N}\Sigma-lnp(y_i|x_i,m,\theta\}$
    - MAP
        - Need to assume prior probability before training
        - Maximize the posterior
        - After input data, use it and the prior to update posterior
        - $\theta_{MAP} = arg max_{\theta}\{lnp(y_i|x_i,m,\theta\}+lnp(\theta|m)$
            - Term $\{lnp(y_i|x_i,m,\theta\}$ is just like MLE
            - $p(\theta|m)$ is the prior
    - Homework note
        - Linear regression can use closed form or gradient descent to find $w$
            - Closed form: $w = (X^TX)^{-1}X^TY$
                - It's a **least square solution**
                - Need a lot of time to calculate when dataset is big
            - Gradient descent: $w^{new} = w^{old} - \alpha\bigtriangledown E(w)$
                - $E(w)$ is error function 
                - $\bigtriangledown E(w)$ is the gradient of error function with respect to w
### HW2
1. Sequential Bayesian Learning
    - Basis Function
        - Assume the number of basis function $\phi(x)$ is M
        - Input a scalar x, you get an array $[x_1,x_2,...,x_M]$
2. Logistic Regression
    - Newton method: $w^{new} = w^{old} + H^{-1}\bigtriangledown E(w)$
    - Batch GD
        - 一次把全部training data的$\bigtriangledown E(w)$加起來，再去更新w
    - Stochastic GD
        - 一次只算一個training data的$\bigtriangledown E(w)$，就更新w
    - Mini-batch GD
        - 一次只算給定的batch size的training data的$\bigtriangledown E(w)$，才更新w
    - Homework note
        - Feature的covariance matrix，可以用np.linalg.eig去拿eigenvector去計算
        - 用eigenvector畫圖的時候不知道會甚麼要用np.linalg.svd裡面的eigenvector才畫得像樣的圖
### HW3
1. Gaussian Process for Regression
2. Support Vector Machine
    - Import libraray`from sklearn.svm import SVC`
    - Function call
        ```python
            # For linear kernel SVM
            linear_svm = SVC(kernel = 'linear')
            # For polynomial(degree of 2) kernel SVM
            poly_svm = SVC(kernel = 'poly',degree = 2)
            # To fit the training data
            linear_svm.fit(train_x,train_t) #注意train_t內的elements只能是-1 or +1
            # Get parameter from model
            support_vectors = linear_svm.support_vectors_
            at = linear_svm.dual_coef_ # 這個是SVM的y(x) function內會用到的langrange coefficient a乘上x的label t的值
            b = linear_svm.intercept_ # 一樣是y(x) function內會用到的bias值
        ```
3. Gaussian Mixture Model
    - Homework note
        - 先用K-means去找到M個中心點，叫做mu
        - 算出pi,mu,sigma
        - 再拿去當作GMM的初始值
        - 再依序用E-step/M-step去更新pi,mu,sigma
            - E-step
                - 求出新的responsibility $\gamma$
            - M-step
                - 根據公式算出新的pi,mu,sigma
