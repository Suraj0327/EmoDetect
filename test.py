import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from tempfile import NamedTemporaryFile
from datetime import datetime
from pymongo import MongoClient
from passlib.hash import bcrypt
from matplotlib import pyplot as plt
import base64
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from bson.binary import Binary
import google.generativeai as genai


def chatbot():
    genai.configure(api_key="AIzaSyCLy3ZU0MfToJLuwqiUrKZGgn8mTj1scr8")  # Replace 'YOUR_API_KEY' with your actual API key
    model = genai.GenerativeModel("gemini-pro") 
    chat = model.start_chat()

    def LLM_Response(question):
        response = chat.send_message(question, stream=True)
        return response

    st.title("How can i help you ?")

    user_quest = st.text_input("Ask a question:")
    btn = st.button("Ask")

    if btn and user_quest:
        st.subheader("Response:")
        result = LLM_Response(user_quest)
        for word in result:
            st.text(word.text)

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://surajmahto0739:d6xdtaeXrD4Z3g0H@cluster0.dp2tmqz.mongodb.net/<dbname>?retryWrites=true&w=majority&appName=Cluster0")  # Replace 'YOUR_MONGODB_CONNECTION_STRING' with your actual connection string
db = client["emodetect"]
users_collection = db["users"]
reports_collection = db["reports"]

# Define the list of emotions to be analyzed
emotions = ['happy', 'angry', 'surprise', 'fear', 'neutral', 'sad']

# Function to register a new user
def register_user(name, email, password, age, gender):
    if users_collection.find_one({"email": email}):
        st.error("User already exists with this email.")
    else:
        hashed_password = bcrypt.hash(password)  # Hash the password
        user_data = {"name": name, "email": email, "password": hashed_password, "age": age, "gender": gender}
        users_collection.insert_one(user_data)
        st.success("User registered successfully.")

# Function to authenticate a user
def authenticate_user(email, password):
    user_data = users_collection.find_one({"email": email})
    if user_data:
        hashed_password = user_data["password"]
        if bcrypt.verify(password, hashed_password):  # Verify the password hash
            return True, user_data["_id"]
    return False, None

# Function to store generated report into MongoDB
def store_report(user_id, timestamp, user_name, depression_score, anxiety_score, emotions_contributing, pdf_report):
    # Convert emotions_contributing DataFrame to dictionary
    emotions_dict = emotions_contributing.to_dict(orient='records')
    
    report_data = {
        "user_id": user_id,
        "timestamp": timestamp,
        "user_info": user_name,
        "depression_score": depression_score,
        "anxiety_score": anxiety_score,
        "emotions_contributing": emotions_dict,
        "pdf_report": Binary(pdf_report)
    }
    reports_collection.insert_one(report_data)
    st.success("Report stored successfully.")

# Function to recommend music based on depression and anxiety scores
def recommend_music(depression_score, anxiety_score):
    if depression_score >= 75 or anxiety_score >= 75:
        return "https://www.youtube.com/results?search_query=calm+music"
    elif depression_score >= 50 or anxiety_score >= 50:
        return "https://www.youtube.com/results?search_query=happy+music"
    else:
        return "https://www.youtube.com/results?search_query=uplifting+music"

# Function to process video frames from a file
def process_video_file(file):
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.read())
        video_path = temp_file.name
        cap = cv2.VideoCapture(video_path)
        process_video_frames(cap)

# Function to process frames from video capture
def process_video_frames(cap):
    st.subheader("Processing Video...")
    total_frames = 0
    depression_score = 0
    anxiety_score = 0
    emotion_df = pd.DataFrame(columns=emotions)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform emotion analysis using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Extract the emotion scores for the first face detected in the image
        emotions_scores = result[0].get('emotion', {})
        
        # Calculate depression score
        depression_score += ((emotions_scores.get('angry', 0) + emotions_scores.get('sad', 0) + 0.5 * emotions_scores.get('fear', 0)) / 
                            ((sum(emotions_scores.values()) or 1))) * 100
        
        # Calculate anxiety score
        anxiety_score += ((emotions_scores.get('fear', 0) + 0.5 * emotions_scores.get('angry', 0) +
                            emotions_scores.get('sad', 0)) / ((sum(emotions_scores.values()) or 1))) * 100

        # Append emotion scores to DataFrame
        emotion_df = pd.concat([emotion_df, pd.DataFrame([emotions_scores], columns=emotions)], ignore_index=True)

        # Increment total frames counter
        total_frames += 1
    
    # Release video capture
    cap.release()

    # Calculate average depression and anxiety scores
    if total_frames != 0:
        depression_score /= total_frames
        anxiety_score /= total_frames

        st.write(f'Average Depression Score: {depression_score:.2f}')
        st.write(f'Average Anxiety Score: {anxiety_score:.2f}')

        # Plot emotion scores using line chart
        st.header('Line Chart for All Emotions')
        st.line_chart(emotion_df)

        # Plot pie diagram
        st.header('Pie Chart')
        fig, ax = plt.subplots()
        ax.pie(emotion_df.mean(), labels=emotions, autopct='%1.1f%%')
        st.pyplot(fig)

        # Display Depression Report
        st.header('Depression Report')
        st.write(f'Depression Score: **{depression_score:.2f}**({("Healthy" if depression_score < 50 else "Moderately Depressed") if depression_score < 75 else "Highly Depressed"})')
        st.write('### Emotions Contributing to Depression:')
        for emotion, score in zip(['Angry', 'Sad', 'Fear'], [emotion_df[emotion].mean() for emotion in ['angry', 'sad', 'fear']]):
            st.write(f"- **{emotion}:** {score:.2f}%")

        # Display Anxiety Report
        st.header('Anxiety Report')
        st.write(f'Anxiety Score: **{anxiety_score:.2f}** ({("Healthy" if anxiety_score < 50 else "Moderately Anxious") if anxiety_score < 75 else "Highly Anxious"})')
        st.write('### Emotions Contributing to Anxiety:')
        for emotion, score in zip(['Fear', 'Angry', 'Sad'], [emotion_df[emotion].mean() for emotion in ['fear', 'angry', 'sad']]):
            st.write(f"- **{emotion}:** {score:.2f}%")

        # Generate PDF report
        generate_pdf_report(depression_score, anxiety_score, emotion_df)
        music_link = recommend_music(depression_score, anxiety_score)
        st.markdown(
            f'<a href="{music_link}" target="_blank" style="background-color: #007bff; color: #ffffff; font-size: 16px; padding: 10px 20px; border-radius: 5px; border: none; cursor: pointer;">Recommended Music</a>',
            unsafe_allow_html=True
        )

# Function to stream video while capturing frames from webcam
def process_webcam_frames(duration):
    st.subheader("Webcam")
    video_stream = st.empty()
    
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    
    # Get the start time
    start_time = time.time()
    elapsed_time = 0
    
    # Initialize variables to store emotion scores and DataFrame to store emotion scores
    total_frames = 0
    depression_score = 0
    anxiety_score = 0
    emotion_df = pd.DataFrame(columns=emotions)
    
    # Capture frames for the specified duration (in seconds)
    while elapsed_time < duration:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam.")
            break
        
        # Display frame in Streamlit app
        video_stream.image(frame, channels="BGR")
        
        # Perform emotion analysis using DeepFace
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Extract the emotion scores for the first face detected in the image
        emotions_scores = result[0].get('emotion', {})
        
        # Calculate depression score
        depression_score += ((emotions_scores.get('angry', 0) + emotions_scores.get('sad', 0) + 0.5 * emotions_scores.get('fear', 0)) / 
                            ((sum(emotions_scores.values()) or 1))) * 100
        
        # Calculate anxiety score
        anxiety_score += ((emotions_scores.get('fear', 0) + 0.5 * emotions_scores.get('angry', 0) +
                            emotions_scores.get('sad', 0)) / ((sum(emotions_scores.values()) or 1))) * 100

        # Append emotion scores to DataFrame
        emotion_df = pd.concat([emotion_df, pd.DataFrame([emotions_scores], columns=emotions)], ignore_index=True)

        # Increment total frames counter
        total_frames += 1
        
        # Update elapsed time
        elapsed_time = time.time() - start_time

    # Release the webcam
    cap.release()

    # Calculate average depression and anxiety scores
    if total_frames != 0:
        depression_score /= total_frames
        anxiety_score /= total_frames

        st.write(f'Average Depression Score: {depression_score:.2f}')
        st.write(f'Average Anxiety Score: {anxiety_score:.2f}')

        # Plot emotion scores using line chart
        st.header('Line Chart for All Emotions')
        st.line_chart(emotion_df)

        # Plot pie diagram
        st.header('Pie Chart')
        fig, ax = plt.subplots()
        ax.pie(emotion_df.mean(), labels=emotions, autopct='%1.1f%%')
        st.pyplot(fig)

        # Display Depression Report
        st.header('Depression Report')
        st.write(f'Depression Score: **{depression_score:.2f}**({("Healthy" if depression_score < 50 else "Moderately Depressed") if depression_score < 75 else "Highly Depressed"})')
        st.write('### Emotions Contributing to Depression:')
        for emotion, score in zip(['Angry', 'Sad', 'Fear'], [emotion_df[emotion].mean() for emotion in ['angry', 'sad', 'fear']]):
            st.write(f"- **{emotion}:** {score:.2f}%")

        # Display Anxiety Report
        st.header('Anxiety Report')
        st.write(f'Anxiety Score: **{anxiety_score:.2f}** ({("Healthy" if anxiety_score < 50 else "Moderately Anxious") if anxiety_score < 75 else "Highly Anxious"})')
        st.write('### Emotions Contributing to Anxiety:')
        for emotion, score in zip(['Fear', 'Angry', 'Sad'], [emotion_df[emotion].mean() for emotion in ['fear', 'angry', 'sad']]):
            st.write(f"- **{emotion}:** {score:.2f}%")
        
        # Generate PDF report
        generate_pdf_report(depression_score, anxiety_score, emotion_df)
        music_link = recommend_music(depression_score, anxiety_score)
        st.markdown(
            f'<a href="{music_link}" target="_blank" style="background-color: #007bff; color: #ffffff; font-size: 16px; padding: 10px 20px; border-radius: 5px; border: none; cursor: pointer;">Recommended Music</a>',
            unsafe_allow_html=True
        )
        chatbot()

# Function to generate PDF report
def generate_pdf_report(depression_score, anxiety_score, emotion_df):
    user_id = st.session_state["user_id"]
    user_data = users_collection.find_one({"_id": user_id})
    if user_data:
        user_name = user_data.get("name", "")
    else:
        user_name = "Unknown"
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Emotion Report")
    c.drawString(100, 730, f"Depression Score: {depression_score:.2f}")
    c.drawString(100, 710, f"Anxiety Score: {anxiety_score:.2f}")

    # Draw line chart
    fig, ax = plt.subplots()
    for emotion in emotions:
        ax.plot(emotion_df[emotion], label=emotion)
    ax.legend(loc='upper right', fontsize='small')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("emotion_line_chart.png")

    # Draw the line chart in the PDF report
    c.drawImage("emotion_line_chart.png", 50, 400, width=500, height=300)

    # Create and save the pie chart as PNG
    fig, ax = plt.subplots()
    ax.pie(emotion_df.mean(), labels=emotions, autopct='%1.1f%%')
    plt.tight_layout()
    plt.savefig("emotion_pie_chart.png")

    # Draw the pie chart in the PDF report
    c.drawImage("emotion_pie_chart.png", 100, 100, width=400, height=300)

    # Draw text report
    text_report = f"Depression Score: {depression_score:.2f}\n"
    text_report += f"Anxiety Score: {anxiety_score:.2f}\n\n"
    text_report += "Emotion Distribution:\n"
    for emotion, score in emotion_df.mean().items():
        text_report += f"{emotion}: {score:.2f}%\n"

    # Draw text report in the PDF
    c.drawString(100, 750, text_report)

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    st.success("PDF report generated successfully.")
    store_report(user_id, datetime.now(), user_name, depression_score, anxiety_score, emotion_df, pdf_bytes)
    st.markdown(get_download_link(pdf_bytes, "emotion_report.pdf"), unsafe_allow_html=True)

# Function to generate download link for PDF
def get_download_link(data, filename):
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download PDF report</a>'
    return href


# Main application
def main():
    st.title('EmoDetect: Facial Assessment of Anxiety and Depression')

    # Check if the user is logged in
    if "user_id" not in st.session_state:
        # If not logged in, show login page
        show_login_page()
    else:
        # If logged in, show main application page based on the selected option
        st.header("Please choose an option to start emotion detection")
        option = st.radio("Choose an option", ("Upload a video file", "Use webcam"))
        if option == "Upload a video file":
            show_video_upload()
        elif option == "Use webcam":
            duration = 30  # Capture frames for 30 seconds
            process_webcam_frames(duration)

    # After the main functionality, call the chatbot
    

# Define function to show login page
def show_login_page():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        authenticated, user_id = authenticate_user(email, password)
        if authenticated:
            st.session_state["user_id"] = user_id
            st.success("Login successful. Redirecting to main application...")
            st.experimental_rerun()
        else:
            st.error("Invalid email or password.")
    st.write("Don't have an account? Register below.")
    show_registration_form()

# Define function to show registration form
def show_registration_form():
    st.title("Registration")
    name = st.text_input("Name")
    email = st.text_input("Email", key="email_input")  # Add unique key
    password = st.text_input("Password", type="password", key="password_input")  # Add unique key
    age = st.number_input("Age", min_value=0, max_value=150, value=18)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    if st.button("Register"):
        register_user(name, email, password, age, gender)

# Define function to show video upload option
def show_video_upload():
    st.subheader("Upload a video file")
    file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if file is not None:
        process_video_file(file)
        st.subheader("Uploaded Video:")
        st.video(file)

# Run the Streamlit app
if __name__ == "__main__":
    main()
