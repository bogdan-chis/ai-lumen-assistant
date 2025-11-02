import pyttsx3

def speak_ro(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.setProperty(
        "voice",
        "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\MSTTS_V110_roRO_Andrei"
    )
    engine.say(text)
    engine.runAndWait()