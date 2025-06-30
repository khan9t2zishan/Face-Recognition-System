# import os
# import cv2
# import numpy as np
# from deepface import DeepFace

# # Create dataset

# dir = "DataSet"
# os.makedirs(dir, exist_ok=True);

# def create_dataset(name):
#     person = os.path.join(dir, name)
#     os.makedirs(person, exist_ok=True)
    
#     cap = cv2.VideoCapture(0)
#     count = 0;
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture image")
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
        
#         for (x, y, w, h) in faces:
#             count += 1
#             face_img = frame[y:y+h, x:x+w]
#             face_path = os.path.join(person, f"{name}_{count}.jpg")
#             cv2.imwrite(face_path, face_img) 
            
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
#         cv2.imshow("Capturing Face in camera", frame)
            
#         if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"Captured {count} images for {name}.")
    

# ## Train face recognition dataset model

# def train_dataset():
#     embeddings = {}  # Initialize the dictionary first
#     for i in os.listdir(dir):
#         person = os.path.join(dir, i)

#         if os.path.isdir(person):
#             embeddings[i] = []  # Fix: Correct variable name
#             for img_name in os.listdir(person):
#                 img_path = os.path.join(person, img_name)
#                 try:
#                     embedding = DeepFace.represent(img_path, model_name='Facenet', enforce_detection=False)[0]["embedding"]
#                     embeddings[i].append(embedding)  # Fix: Correct variable name
#                 except Exception as e:
#                     print(f"Failed to train images {img_path}: {e}")
#     return embeddings


# ## Recognize faces, age, gender, emotion usinng DeepFace

# def recognize_Face(embeddings):
#     cap = cv2.VideoCapture(0)
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture image")
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5) 
        
#         for (x, y, w, h) in faces:
#             face_img = frame[y:y+h, x:x+w]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
#             try:
#                 analyse  = DeepFace.analyze(face_img, actions = ["age", "gender", "emotion"], enforce_detection=False)
            
#                 if isinstance(analyse, list):
#                     analyse = analyse[0]
                
#                 age = analyse["age"]
#                 gender = analyse["gender"]
#                 gender = gender if isinstance(gender, str) else max(gender, key=gender.get)
#                 emotion = max(analyse["emotion"], key=analyse["emotion"].get)
            
#                 face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            
#                 match = None
#                 max_similarity = -1
            
#                 for person, person_embeddings in embeddings.items():
#                     for embedding in person_embeddings:
#                         similarity = np.dot(face_embedding, embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(embedding))
#                         if similarity > max_similarity:
#                             max_similarity = similarity
#                             match = person
                        
#                 if max_similarity > 0.7:  
#                     label = f"{match}({max_similarity:.2f})"
#                 else:
#                     label = "Unknown Person"
            
#                 display_text = f"{label}, Age:{int(age)}, Gender:{gender}"
#                 # , Emotion:{emotion}
#                 cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)    
            
#             except Exception as e:
#                 print(f"Error in face recognition: {e}")
    
#         cv2.imshow("Face Recognition", frame)
    
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cap.destroyAllWindows()
    
# # Output

# if __name__ == "__main__":
#     print("Welcome to Face Recognition System")
#     print("1. Create a new dataset")
#     print("2. Train dataset")  
#     print("3. Recognize faces in real-time")
#     choice = input("Enter your choice (1/2/3): ")
    
#     if choice == '1':
#         name = input("Enter your name: ")
#         create_dataset(name)
        
#     elif choice == '2':
#         embedding = train_dataset()
#         np.save("embedding.npy", embedding)
#         print("Dataset trained successfully.")
        
#     elif choice == '3':
#         if os.path.exists("embedding.npy"):
#             embedding = np.load("embedding.npy", allow_pickle=True).item()
#             # print("Please train the dataset first.")
#             recognize_Face(embedding)
            
#         else:
#             print("No trained dataset found. Please train the dataset first.")
        
#     else:
#         print("Invalid choice. Please try again.")
    
    
            
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# Create dataset
dir = "DataSet"
os.makedirs(dir, exist_ok=True)

def create_dataset(name):
    person = os.path.join(dir, name)
    os.makedirs(person, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y+h, x:x+w]
            face_path = os.path.join(person, f"{name}_{count}.jpg")
            cv2.imwrite(face_path, face_img) 
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
        cv2.imshow("Capturing Face in Camera", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for {name}.")

# Train face recognition dataset model
def train_dataset():
    embeddings = {}
    for i in os.listdir(dir):
        person = os.path.join(dir, i)
        if os.path.isdir(person):
            embeddings[i] = []
            for img_name in os.listdir(person):
                img_path = os.path.join(person, img_name)
                try:
                    embedding = DeepFace.represent(img_path, model_name='Facenet', enforce_detection=False)[0]["embedding"]
                    embeddings[i].append(embedding)
                except Exception as e:
                    print(f"Failed to train images {img_path}: {e}")
    return embeddings

# Function to maintain Excel log of recognized faces
import pandas as pd
from datetime import datetime

def update_excel(person_name):
    file_name = "Face_Recognition_Log.xlsx"
    today = datetime.today().strftime("%Y-%m-%d")

    try:
        # Try reading the existing file and sheet
        with pd.ExcelFile(file_name) as xls:
            if today in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=today)
            else:
                df = pd.DataFrame(columns=["Name", "Date", "Time"])
    except FileNotFoundError:
        # If file doesn't exist, create a new DataFrame
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    now = datetime.now().strftime("%H:%M:%S")

    # Check if the person is already logged today
    if person_name in df["Name"].values:
        print(f"{person_name} is already recorded today. Skipping entry.")
        return  # Stop execution if the entry exists

    # If the person is not recorded, add them
    new_entry = {"Name": person_name, "Date": today, "Time": now}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

    # Save the file with the date-wise sheet
    with pd.ExcelWriter(file_name, mode="w") as writer:
        df.to_excel(writer, sheet_name=today, index=False)

    print(f"{person_name} recorded successfully for {today}.")



# Recognize faces and log results
def recognize_Face(embeddings):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5) 
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            try:
                face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                match = None
                max_similarity = -1
            
                for person, person_embeddings in embeddings.items():
                    for embedding in person_embeddings:
                        similarity = np.dot(face_embedding, embedding) / (np.linalg.norm(face_embedding) * np.linalg.norm(embedding))
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match = person
                        
                if max_similarity > 0.7:
                    label = f"{match}({max_similarity:.2f})"
                    update_excel(match)  # Log recognized person
                else:
                    label = "Unknown Person"
            
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green text
            
            except Exception as e:
                print(f"Error in face recognition: {e}")
    
        cv2.imshow("Face Recognition", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Execution flow
if __name__ == "__main__":
    print("Welcome to Face Recognition System")
    print("1. Create a new dataset")
    print("2. Train dataset")  
    print("3. Recognize faces in real-time")
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1':
        name = input("Enter your name: ")
        create_dataset(name)
        
    elif choice == '2':
        embedding = train_dataset()
        np.save("embedding.npy", embedding)
        print("Dataset trained successfully.")
        
    elif choice == '3':
        if os.path.exists("embedding.npy"):
            embedding = np.load("embedding.npy", allow_pickle=True).item()
            recognize_Face(embedding)
        else:
            print("No trained dataset found. Please train the dataset first.")
        
    else:
        print("Invalid choice. Please try again.")
