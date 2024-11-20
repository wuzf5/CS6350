# Perceptron

Dear TA,

I made a small mistake in my previous implementation for not incorporating the bias term. The new implementation gives a slightly different result: 

The test errors of AveragedPerceptron, StandardPerceptron, and VotedPerceptron are 0.014, 0.016, and 0.014, respectively.

The learned weight vectors are [-406358.35084101, -253467.37641, -262862.67272, -77975.50382501, 317252.] and [-56.455303, -44.00955, -38.983458, -5.995289, 61.] respectively for AveragedPerceptron and StandardPerceptron.

Note that the test error and the learned weight vectors still differ in different runs for StandardPerceptron due to the randomness resulting from shuffling.

I apologize for the trouble and will be grateful if the TA can reconsider the grades for this Homework.

Best,

Zifan


## Usage

For StandardPerceptron:
```bash
cd ./Perceptron/StandardPerceptron
chmod +x ./run.sh
./run.sh
```
For AveragedPerceptron:
```bash
cd ./Perceptron/AveragedPerceptron
chmod +x ./run.sh
./run.sh
```
For VotedPerceptron:
```bash
cd ./Perceptron/VotedPerceptron
chmod +x ./run.sh
./run.sh
```
