import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from ucimlrepo import fetch_ucirepo
import pickle

# Φορτώνεται το Dataset σύμφωνα με τις οδηγίες που βρίσκονται στο https://archive.ics.uci.edu/dataset/19/car+evaluation
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets

# Ενώνουμε τα χαρακτηριστικά της βάσης με το ζητούμενο στόχο
df = pd.concat([X, y], axis=1)

# Μετατρέπουμε τα κατηγορικά δεδομένα σε αριθμητικές τιμές
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop("class", axis=1)
y = df["class"]

# Χωρίζουμε το dataset μας σε Train/Test. 80% για train και 20% για test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Γίνεται η εκπαίδευση με Random Forest, φτιάχνοντας 100 decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Αξιολογείται το μοντέλο μας
y_pred = model.predict(X_test)
print("Αποτελέσματα Μοντέλου")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Παίρνουμε τις προβλέψεις που κάνει το μοντέλο μας από το test set
y_pred = model.predict(X_test)

# Φτιάχνουμε ένα array παίρνοντας ως input τα πραγματικά αποτελέσματα και τις προβλέψεις του μοντέλου μας
cm = confusion_matrix(y_test, y_pred)

# Αποθηκεύουμε το matrix και τα test results
results = {
    "y_test": y_test,
    "y_pred": y_pred,
    "cm": cm
}

# Αποθηκεύουμε τα αποτελέσματα αξιολόγησης του μοντέλου μας
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

# Αποθηκεύουμε το μοντέλο μας
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
# Αποθηκεύουμε τους encoders
with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
