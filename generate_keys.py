import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["Mirsha Morningstar", "Rameez Akther", "Chandru", "Mekesh"]
usernames = ["Mirsha Morningstar","Rameez", "Chandru", "Mekesh"]

passwords = ["AKM69", "Ramzy123", "3.14159", "mekesh123"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent/"hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords,file)