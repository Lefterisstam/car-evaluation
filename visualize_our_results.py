import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Φορτώνουμε τα αποθηκευμένα αποτελέσματα
with open("results.pkl", "rb") as f:
    results = pickle.load(f)

y_test = results["y_test"]
y_pred = results["y_pred"]
cm = results["cm"]

# Εμφανίζουμε το matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (Test Set)")
plt.show()
