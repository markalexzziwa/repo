import streamlit as st
import datetime
import random

# Page configuration
st.set_page_config(
    page_title="Simple Greeting App",
    page_icon="👋",
    layout="centered"
)

# Main app
def main():
    # Header with greeting
    st.title("👋 Welcome to Our Greeting App")
    
    # Get current time for dynamic greeting
    current_hour = datetime.datetime.now().hour
    
    if current_hour < 12:
        greeting = "Good morning! ☀️"
    elif current_hour < 18:
        greeting = "Good afternoon! 🌤️"
    else:
        greeting = "Good evening! 🌙"
    
    # Display greeting
    st.subheader(greeting)
    
    # User input section
    st.write("---")
    
    # User name input
    name = st.text_input("What's your name?", placeholder="Enter your name here...")
    
    if name:
        st.success(f"Hello {name}! Nice to meet you! 👋")
        
        # Mood selection
        st.write("How are you feeling today?")
        mood = st.selectbox(
            "Select your mood:",
            ["Great! 😄", "Good 🙂", "Okay 😐", "Tired 😴", "Excited! 🎉"],
            key="mood"
        )
        
        if st.button("Send Greeting"):
            st.balloons()
            st.info(f"Hello {name}! You're feeling {mood.split(' ')[0].lower()} today!")

    # Simple interactive section
    st.write("---")
    
    if st.button("Show Random Greeting"):
        greetings = [
            "Have a wonderful day! 🌟",
            "You're amazing! ✨",
            "Keep smiling! 😊",
            "Make today great! 🚀",
            "Sending positive vibes! 💫"
        ]
        st.success(random.choice(greetings))

if __name__ == "__main__":
    main()