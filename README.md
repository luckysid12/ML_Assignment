# ML_Assignment

Name: Siddhartha Kanugu

Student ID: 700772579

CRN: 11438

1 — Function Approximation by Hand

Dataset: (x,y)={(1,1),(2,2),(3,2),(4,5)}

Task:

1.	Try model θ=(1,0).Fill in predictions, residuals, squared residuals, and compute MSE.

2.	Try model θ=(0.5,1). Do the same.

3.	Which model fits better?

Dataset: (x,y)={(1,1),(2,2),(3,2),(4,5)}

Model A: θ=(θ0,θ1)=(1,0)

Predictions y^: 1, 1, 1, 1

Residuals r=y−y^: 0, 1, 1, 4

Squared residuals: 0, 1, 1, 16

MSE =0+1+1+16/4=4.5

Model B: θ=(0.5,1)

y^: 1.5, 2.5, 3.5, 4.5

r: −0.5, −0.5, −1.5, 0.5

r2: 0.25, 0.25, 2.25, 0.25

MSE =3.0/4=0.75

Better fit: Model B (lower MSE = 0.75).

2 — Random Guessing Practice

Cost function: J(θ_1, θ_2)=8〖(θ_1-0.3)〗^2+4〖(θ_2-0.7)〗^2

Task:

1. Compute J(0.1,0.2) and J(0.5,0.9).

2. Which guess is closer to the minimum (0.3,0.7)?

3. In 2–3 sentences, explain why random guessing is inefficient.

Compute:

•	J(0.1,0.2)=8(−0.2)2+4(−0.5)2=8(0.04)+4(0.25)=0.32+1=1.32

•	J(0.5,0.9)=8(0.2)2+4(0.2)2=8(0.04)+4(0.04)=0.32+0.16=0.48

Closer to the minimum (0.3,0.7): (0.5,0.9).

Why random guessing is inefficient (2–3 sentences):

The search space is continuous, so random picks rarely land near the minimizer. Each guess ignores curvature/gradient information, so progress is slow and inconsistent. Gradient-based updates systematically move downhill and converge much faster.

3 — First Gradient Descent Iteration

Dataset: (1,3),(2,4),(3,6),(4,5)

Start: θ^(0)=(0,0), learning rate α=0.01.

Task:

	Compute predictions at θ^(0).

	Compute residuals and sums ∑r, ∑xr.

	Compute gradient ∇J.

	Update to θ^(1).

	Compute J(θ^(0)) and J(θ^(1)).

Continue from Homework 3 with θ^(1)

Task:

	Compute predictions at θ^(1).

	Compute residuals, ∑r, ∑xr .

	Compute gradient.

	Update to θ^((2)).

	Compare J(θ^(1)) and J(θ^(1)).

Use J(θ)=1/2m∑ri2 with m=4

Then

∇J(θ)=(−1/m∑r,  −1/m∑xr)

Step 0 (at θ^(0)=(0,0))

Predictions y^: 0, 0, 0, 0

Residuals r: 3, 4, 6, 5

∑r=18,  ∑xr=49

Gradient ∇J=(−4.5, −12.25)

Update:

θ^(1)=θ^(0)−α∇J=(0,0)−0.01(−4.5,−12.25)=(0.045,  0.1225)

Costs:

J(θ^(0))=18(9+16+36+25)=10.75

J(θ^(1))≈9.117942

Step 1 (from θ^(1)=(0.045,0.1225))

Preds y^: 0.1675, 0.29, 0.4125, 0.535

Residuals r: 2.8325, 3.71, 5.5875, 4.465

∑r=16.595,  ∑xr=44.875

∇J=(−4.14875, −11.21875)

Update:

θ^(2)=(0.0864875,  0.2346875)

Costs:

J(θ^(1))≈9.117942 → J(θ^(2))≈7.746912 (decreased).

4 — Compare Random Guessing vs Gradient Descent

Dataset: (1,2),(2,2),(3,4),(4,6)

Task:

	Try 2 random guesses: (θ_1, θ_2)=(0.2,0.5) and (0.9,0.1). Compute J.

	Compare with J from the first gradient descent step (starting at θ=(0,0), α=0.01).

	Which approach gave lower error? Why?

Dataset: (1,2),(2,2),(3,4),(4,6), J=1/2m∑r^2

Random guesses

•	θ=(0.2,0.5):  J=2.7575

•	θ=(0.9,0.1):  J=3.9675

One GD step from θ=(0,0), α=0.01:

∑r=14,  ∑xr=42⇒∇J=(−3.5,−10.5)

θ=(0.035,0.105), J drops from 7.5 to 6.326144.

Which is lower? The random guess (0.2,0.5) (2.7575) is lower than GD after only
one step. But GD is systematic; with more steps it will keep reducing J and eventually beat most random guesses.

5 — Recognizing Underfitting and Overfitting

Imagine you train a model and notice the following results:

•	Training error is very high.

•	Test error is also very high.

Questions:

1.	Is this an example of underfitting or overfitting?

2.	Explain why this situation happens.

3.	Suggest two possible fixes.

Training error high, test error high → Underfitting.

Why: Model is too simple, under-trained, or features are inadequate; it can’t capture the signal.

Fixes (any two): Increase model capacity/features; reduce regularization; train longer; improve data quality/labels.

6 — Comparing Models

You test two different machine learning models on the same dataset:

•	Model A fits the training data almost perfectly but performs poorly on new unseen data.

•	Model B does not fit the training data very well and also performs poorly on unseen data.

Questions:

1.	Which model is overfitting? Which one is underfitting?

2.	In each case, what is the tradeoff between bias and variance?

3.	What would you recommend to improve each model?

Model A (near-perfect train, poor test) → Overfitting (low bias, high variance).

Improve: regularization (L2/L1), simplify architecture, collect more data/augmentation, early stopping, cross-validation.

Model B (poor train & test) → Underfitting (high bias, low variance).

Improve: add features, increase capacity, reduce regularization, train longer, tune learning rate.

7 —Programming Problem - Implement Gradient Descent for Linear Regression

Problem Statement

You are asked to implement linear regression using Gradient Descent from scratch (without using scikit-learn’s LinearRegression). Your task is to compare the closed-form solution (Normal Equation) with Gradient Descent on the same dataset.

Dataset

•	Generate synthetic data following: y=3+4x+ϵ where ϵ is Gaussian noise.

•	Create 200 samples with x∈[0,5].

Requirements

1.	Generate the dataset and plot the raw data.

2.	Closed-form solution (Normal Equation):

    o	Compute  
  
    o	Print the estimated intercept and slope.
  
    o	Plot the fitted line.

3.	Gradient Descent implementation:
  
    o	Initialize θ=[0,0].
  
    o	Use learning rate η=0.05.
  
    o	Run for 1000 iterations.
  
    o	Plot the loss curve (MSE vs iterations).
  
    o	Print the final intercept and slope.

4.	Comparison:

    o	Report both solutions (Closed-form vs Gradient Descent).
  
    o	Comment: Did Gradient Descent converge to the same solution as the closed form?

Expected Deliverables

•	Python code file (.py or Jupyter notebook).

•	A plot showing:

    o	Raw data points.

    o	Closed-form fitted line.

    o	Gradient Descent fitted line.

•	A plot of loss vs iterations (for Gradient Descent).

•	A short explanation (2–3 sentences) of the results.

Hints

•	Don’t forget to add a bias column of 1’s to your X.

•	The gradient for MSE is:

•	Use np.dot() for matrix multiplication.

•	To plot multiple lines on the same figure, use plt.plot() several times.
