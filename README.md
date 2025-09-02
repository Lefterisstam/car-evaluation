# Car Evaluation Prediction (Random Forest)

This project is a Machine Learning application that predicts the evaluation of a car (e.g., acceptable, good, very good, etc.) using the [Car Evaluation dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation) from the UCI Machine Learning Repository.  

We train a Random Forest Classifier, visualize the results with a Confusion Matrix, and provide an interactive Streamlit interface for predictions.



## Requirements + How to run the app

Install dependencies with:

```bash
pip install -r requirements.txt
```

Train our model:

```bash
python train_our_model.py
```

Visualize our results:

```bash
python visualize_our_results.py
```

Start web interface:

```bash
streamlit run app.py
```
