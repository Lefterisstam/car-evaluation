import pickle
import streamlit as st

# Φορτώνουμε το μοντέλο μαζί με τους encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Δημιουργούμε ένα απλό UI με Streamlit, όπου μπορεί κανείς να επιλέξει τιμές για κάθε χαρακτηριστικό
st.title("Car Evaluation Prediction (Random Forest)")

buying = st.selectbox("Buying Price", label_encoders["buying"].classes_)
maint = st.selectbox("Maintenance Cost", label_encoders["maint"].classes_)
doors = st.selectbox("Number of Doors", label_encoders["doors"].classes_)
persons = st.selectbox("Capacity (persons)", label_encoders["persons"].classes_)
lug_boot = st.selectbox("Luggage Boot Size", label_encoders["lug_boot"].classes_)
safety = st.selectbox("Safety", label_encoders["safety"].classes_)

# Μετατρέπουμε τις κατηγορικές τιμές σε αριθμητικές
input_data = [
    label_encoders["buying"].transform([buying])[0],
    label_encoders["maint"].transform([maint])[0],
    label_encoders["doors"].transform([doors])[0],
    label_encoders["persons"].transform([persons])[0],
    label_encoders["lug_boot"].transform([lug_boot])[0],
    label_encoders["safety"].transform([safety])[0]
]

# Κάνουμε την πρόβλεψη και εμφανίζουμε το αποτέλεσμα
if st.button("Πρόβλεψη"):
    prediction = model.predict([input_data])[0]
    prediction_label = label_encoders["class"].inverse_transform([prediction])[0]
    st.success(f"Πρόβλεψη: {prediction_label}")