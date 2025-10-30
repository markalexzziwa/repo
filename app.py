import streamlit as st
import datetime

# Page configuration
st.set_page_config(
    page_title="Greeting App",
    page_icon="👋",
    layout="centered"
)

# Main app
def main():
    # Header with greeting
    st.title("🎉 Welcome to My Streamlit App!")
    
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
    st.header("Let's get to know each other!")
    
    # User name input
    name = st.text_input("What's your name?", placeholder="Enter your name here...")
    
    if name:
        st.success(f"Hello {name}! Nice to meet you! 👋")
        
        # Additional personalized greeting
        col1, col2 = st.columns(2)
        
        with col1:
            mood = st.selectbox(
                f"How are you feeling today, {name}?",
                ["Great! 😄", "Good 🙂", "Okay 😐", "Could be better 😔", "Excited! 🎉"]
            )
        
        with col2:
            if st.button("Send Greeting"):
                st.balloons()
                st.success(f"Wishing you a wonderful day, {name}! You're feeling {mood.split(' ')[0].lower()}")

    # Features section
    st.write("---")
    st.header("App Features")
    
    features = [
        "✨ Friendly greeting interface",
        "🎨 Beautiful Streamlit design",
        "⏰ Time-based greetings",
        "📱 Responsive layout",
        "🎈 Interactive elements"
    ]
    
    for feature in features:
        st.write(feature)

    # Footer
    st.write("---")
    st.markdown("---")
    st.markdown("### Thanks for visiting! Have a great day! 🎊")

if __name__ == "__main__":
    main()