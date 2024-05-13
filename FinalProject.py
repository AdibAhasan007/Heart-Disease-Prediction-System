import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, font
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import threading
import pickle  # To save and load the trained model

data_path = r"D:\Virsity Files\ML\Final Project\Heart Disease Prediction\Heart Disease Prediction\heart.csv"
data = pd.read_csv(data_path)
scaler = StandardScaler()

def train_model():
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train_scaled, y_train)

    # Saving the trained model
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(grid_search.best_estimator_, file)

    messagebox.showinfo("Training Complete", "The model has been trained and saved successfully!")

def load_model():
    global model  # Global declaration to use this model in prediction
    with open('trained_model.pkl', 'rb') as file:
        model = pickle.load(file)
    messagebox.showinfo("Model Loaded", "The model has been loaded successfully!")

def get_user_input():
    new_window = tk.Toplevel(root)
    new_window.title("Patient Data Input")
    new_window.configure(background='light gray')
    entries = {}
    for idx, column in enumerate(data.columns[:-1]):  # Exclude 'target' column
        label = tk.Label(new_window, text=column + ":", bg='light gray', font=('Helvetica', 10, 'bold'))
        label.grid(row=idx, column=0, padx=10, pady=5, sticky='w')
        entry = tk.Entry(new_window, width=25, font=('Helvetica', 10))
        entry.grid(row=idx, column=1, padx=10, pady=5)
        entries[column] = entry

    def predict():
        input_data = [float(entries[col].get()) for col in data.columns[:-1]]
        input_df = pd.DataFrame([input_data], columns=data.columns[:-1])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        disease_status = "has heart disease" if prediction[0] == 1 else "does not have heart disease"
        messagebox.showinfo("Prediction", f"Based on the model, the patient {disease_status}.")
        # Append new data to CSV
        input_data.append(int(prediction[0]))  # Append the prediction as target
        new_df = pd.DataFrame([input_data], columns=data.columns)
        new_df.to_csv(data_path, mode='a', header=False, index=False)
        new_window.destroy()

    predict_btn = tk.Button(new_window, text="Predict", command=predict, font=('Helvetica', 10, 'bold'))
    predict_btn.grid(row=len(data.columns), column=0, columnspan=2, pady=20)

root = tk.Tk()
root.title('Heart Disease Prediction System by Adib Ahasan Chowdhury')
root.configure(background='light blue')

title_font = font.Font(family='Helvetica', size=16, weight='bold')
normal_font = font.Font(family='Helvetica', size=12)

title_label = tk.Label(root, text="Heart Disease Prediction System", font=title_font, bg='light blue')
title_label.pack(pady=10)

train_btn = tk.Button(root, text="Train Model", command=lambda: threading.Thread(target=train_model).start(), font=normal_font, padx=10, pady=5)
train_btn.pack(pady=10)

load_btn = tk.Button(root, text="Load Model", command=load_model, font=normal_font, padx=10, pady=5)
load_btn.pack(pady=10)

input_predict_btn = tk.Button(root, text="Input & Predict", command=get_user_input, font=normal_font, padx=10, pady=5)
input_predict_btn.pack(pady=10)

# Add author label
author_label = tk.Label(root, text="Developed by Adib Ahasan Chowdhury", font=('Helvetica', 10, 'italic'), bg='light blue')
author_label.pack(side='bottom', pady=5)

root.mainloop()
