import pyttsx3

_engine = pyttsx3.init()

def speak(text):
    _engine.say(text)
    _engine.runAndWait()
