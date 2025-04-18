# Enhanced HTML templates for the Streamlit PDF Chat App

css = '''
<style>
/* Main container styles */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Chat message container */
.chat-message {
    padding: 1.5rem;
    border-radius: 0.8rem;
    margin-bottom: 1.5rem;
    display: flex;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

/* User message styling */
.chat-message.user {
    background-color: #2b313e;
    border-left: 5px solid #4d9bff;
}

/* Bot message styling */
.chat-message.bot {
    background-color: #475063;
    border-left: 5px solid #16c97f;
}

/* Avatar container */
.chat-message .avatar {
    width: 10%;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Avatar image */
.chat-message .avatar img {
    max-width: 60px;
    max-height: 60px;
    border-radius: 50%;
    object-fit: cover;
    filter: drop-shadow(0 3px 5px rgba(0, 0, 0, 0.2));
}

/* Message text container */
.chat-message .message {
    width: 90%;
    padding: 0 1.5rem;
    color: #fff;
    line-height: 1.6;
}

/* Code block styling within messages */
.chat-message .message pre {
    background-color: #1e222e !important;
    padding: 1rem;
    border-radius: 0.5rem;
    overflow-x: auto;
    border-left: 3px solid #16c97f;
}

.chat-message .message code {
    font-size: 0.9rem !important;
}

/* App header styling */
h1, h2, h3 {
    color: #4d9bff;
    margin-bottom: 1.5rem;
}

/* File uploader styling */
.stFileUploader > div {
    padding: 1rem;
    border: 2px dashed #4d9bff;
    border-radius: 0.5rem;
}

/* Button styling */
.stButton button {
    background-color: #4d9bff;
    color: white;
    font-weight: bold;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    border: none;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.stButton button:hover {
    background-color: #3a7bda;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transform: translateY(-1px);
}

/* Text input styling */
.stTextInput input {
    border-radius: 0.5rem;
    border: 1px solid #4d9bff;
    padding: 0.75rem;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #2b313e;
}

.sidebar .sidebar-content {
    background-color: #2b313e;
}

/* Streamlit selectbox styling */
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 0.5rem;
    border-color: #4d9bff;
}

/* Success message styling */
.stSuccess {
    background-color: rgba(22, 201, 127, 0.2);
    border-left-color: #16c97f;
}

/* Warning message styling */
.stWarning {
    background-color: rgba(255, 196, 0, 0.2);
    border-left-color: #ffc400;
}

/* Error message styling */
.stError {
    background-color: rgba(255, 88, 88, 0.2);
    border-left-color: #ff5858;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/V3nDgzT/robot-assistant.png" alt="AI Assistant">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/D7g1XVt/user-avatar.png" alt="User">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''