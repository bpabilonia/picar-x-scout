#!/usr/bin/env python3
"""
Roaming Object Detector for PiCar-X

This script enables the PiCar-X to:
1. Roam around the room autonomously with caution (obstacle avoidance)
2. Recognize dogs, cats, and physical objects using OpenAI's multimodal model
3. Stop when detecting objects and speak their names
4. Have conversations about detected objects
5. Support voice wake words: "Stop" to stop, "Start" to resume roaming

Usage:
    sudo python3 roaming_object_detector.py
    sudo python3 roaming_object_detector.py --keyboard
"""

from openai_helper import OpenAiHelper
from keys import OPENAI_API_KEY, OPENAI_ASSISTANT_ID
from preset_actions import actions_dict, sounds_dict
from utils import (
    gray_print, warn, error, 
    redirect_error_2_null, cancel_redirect_error, 
    sox_volume, speak_block
)

import speech_recognition as sr
from openai import OpenAI

from picarx import Picarx
from robot_hat import Music, Pin

import time
import threading
import random
import os
import sys
import base64

os.popen("pinctrl set 20 op dh")  # enable robot_hat speaker switch
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)  # change working directory

# ============================================================================
# Configuration
# ============================================================================
ROBOT_NAME = "Scout"

# Input mode
input_mode = 'voice'
args = sys.argv[1:]
if '--keyboard' in args:
    input_mode = 'keyboard'

# Roaming parameters
ROAM_SPEED = 30  # Forward speed (0-100)
TURN_SPEED = 40  # Turning speed
BACKUP_SPEED = 25  # Reverse speed
MIN_OBSTACLE_DISTANCE = 35  # cm - stop/turn when obstacle is closer
CRITICAL_DISTANCE = 15  # cm - immediate stop and reverse
CAR_WIDTH = 45  # cm - car is about 1.5ft (45cm) wide, need this much clearance
SAFE_CLEARANCE = 60  # cm - comfortable clearance to navigate around obstacles

# Object detection parameters
DETECTION_INTERVAL = 2.0  # seconds between object detection scans
OBJECTS_OF_INTEREST = ['dog', 'cat', 'person', 'chair', 'table', 'ball', 'toy', 'bottle', 'cup', 'phone', 'shoe', 'bag']

# Audio settings
LANGUAGE = ['en']
VOLUME_DB = 3
TTS_VOICE = 'echo'
VOICE_INSTRUCTIONS = "Speak with excitement when you see animals, be curious and friendly."

# Wake words for control
WAKE_WORD_STOP = ['stop', 'freeze', 'halt']
WAKE_WORD_START = ['start', 'go', 'move', 'roam']

# ============================================================================
# OpenAI Helper Extended with Vision
# ============================================================================
class VisionHelper:
    """Helper class for OpenAI Vision API to detect objects in images."""
    
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, timeout=30)
    
    def encode_image_to_base64(self, image_path):
        """Encode image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def detect_objects(self, image_path):
        """
        Analyze image to detect dogs, cats, and physical objects.
        Returns dict with detected objects and descriptions.
        """
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an object detection assistant for a robot car. 
                        Analyze the image and identify any dogs, cats, or physical objects.
                        
                        Respond in this exact JSON format:
                        {
                            "detected": true/false,
                            "objects": [
                                {"name": "object name", "type": "dog/cat/object", "description": "brief description"}
                            ],
                            "summary": "A natural sentence describing what you see",
                            "should_stop": true/false,
                            "conversation_starter": "A friendly question or comment about what you see"
                        }
                        
                        Set "should_stop" to true if you detect a dog, cat, or any interesting object in the immediate path.
                        The robot should stop to observe and interact with living things or notable objects."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What do you see in this image? Look for dogs, cats, people, and any physical objects. Describe what's in view."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            # Try to parse JSON from response
            try:
                import json
                # Find JSON in the response
                start = result_text.find('{')
                end = result_text.rfind('}') + 1
                if start != -1 and end > start:
                    result = json.loads(result_text[start:end])
                    return result
            except:
                pass
            
            # Fallback if JSON parsing fails
            return {
                "detected": False,
                "objects": [],
                "summary": result_text,
                "should_stop": False,
                "conversation_starter": ""
            }
            
        except Exception as e:
            print(f"Vision detection error: {e}")
            return {
                "detected": False,
                "objects": [],
                "summary": "",
                "should_stop": False,
                "conversation_starter": ""
            }

# ============================================================================
# Initialize Components
# ============================================================================
print(f"\n{'='*60}")
print(f"  {ROBOT_NAME} - Roaming Object Detector")
print(f"  Say 'Stop' to stop, 'Start' to roam")
print(f"{'='*60}\n")

# OpenAI helpers
openai_helper = OpenAiHelper(OPENAI_API_KEY, OPENAI_ASSISTANT_ID, 'picarx')
vision_helper = VisionHelper(OPENAI_API_KEY)

# Car initialization
try:
    my_car = Picarx()
    time.sleep(1)
except Exception as e:
    raise RuntimeError(f"Failed to initialize PiCar-X: {e}")

music = Music()
led = Pin('LED')

# Camera initialization
from vilib import Vilib
import cv2

Vilib.camera_start(vflip=False, hflip=False)
Vilib.show_fps()
Vilib.display(local=False, web=True)

while True:
    if Vilib.flask_start:
        break
    time.sleep(0.01)

time.sleep(.5)
print("Camera initialized.\n")

# Speech recognition
recognizer = sr.Recognizer()
recognizer.dynamic_energy_adjustment_damping = 0.16
recognizer.dynamic_energy_ratio = 1.6

# ============================================================================
# State Management
# ============================================================================
class RobotState:
    """Thread-safe state management for the robot."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self._roaming = False
        self._speaking = False
        self._detecting = False
        self._last_detection = None
        self._conversation_mode = False
    
    @property
    def roaming(self):
        with self.lock:
            return self._roaming
    
    @roaming.setter
    def roaming(self, value):
        with self.lock:
            self._roaming = value
    
    @property
    def speaking(self):
        with self.lock:
            return self._speaking
    
    @speaking.setter
    def speaking(self, value):
        with self.lock:
            self._speaking = value
    
    @property
    def conversation_mode(self):
        with self.lock:
            return self._conversation_mode
    
    @conversation_mode.setter
    def conversation_mode(self, value):
        with self.lock:
            self._conversation_mode = value
    
    @property
    def last_detection(self):
        with self.lock:
            return self._last_detection
    
    @last_detection.setter
    def last_detection(self, value):
        with self.lock:
            self._last_detection = value

state = RobotState()

# ============================================================================
# TTS Handler
# ============================================================================
tts_queue = []
tts_lock = threading.Lock()
tts_file = None

def speak(text, blocking=True):
    """Convert text to speech and play it."""
    global tts_file
    
    if not text or len(text.strip()) == 0:
        return
    
    state.speaking = True
    
    try:
        _time = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
        _tts_raw = f"./tts/{_time}_raw.wav"
        _tts_final = f"./tts/{_time}_{VOLUME_DB}dB.wav"
        
        # Generate TTS
        success = openai_helper.text_to_speech(
            text, _tts_raw, TTS_VOICE, 
            response_format='wav', 
            instructions=VOICE_INSTRUCTIONS
        )
        
        if success:
            # Adjust volume
            if sox_volume(_tts_raw, _tts_final, VOLUME_DB):
                tts_file = _tts_final
                speak_block(music, tts_file)
    except Exception as e:
        print(f"TTS error: {e}")
    finally:
        state.speaking = False

# ============================================================================
# Voice Command Detection
# ============================================================================
def listen_for_command(timeout=3):
    """Listen for voice commands. Returns the recognized text or None."""
    _stderr_back = None
    try:
        _stderr_back = redirect_error_2_null()
        with sr.Microphone(chunk_size=8192) as source:
            cancel_redirect_error(_stderr_back)
            _stderr_back = None  # Mark as restored so finally doesn't double-restore
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
            except sr.WaitTimeoutError:
                return None
        
        # Use OpenAI Whisper for STT
        result = openai_helper.stt(audio, language=LANGUAGE)
        return result if result else None
        
    except Exception as e:
        return None
    finally:
        # Ensure stderr is restored if it was redirected but not yet restored
        if _stderr_back is not None:
            cancel_redirect_error(_stderr_back)

def check_wake_words(text):
    """Check if text contains wake words. Returns 'stop', 'start', or None."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    for word in WAKE_WORD_STOP:
        if word in text_lower:
            return 'stop'
    
    for word in WAKE_WORD_START:
        if word in text_lower:
            return 'start'
    
    return None

# ============================================================================
# Object Detection Thread
# ============================================================================
def object_detection_loop():
    """Continuously monitor camera for objects of interest."""
    last_detection_time = 0
    
    while True:
        if not state.roaming or state.speaking:
            time.sleep(0.5)
            continue
        
        current_time = time.time()
        if current_time - last_detection_time < DETECTION_INTERVAL:
            time.sleep(0.2)
            continue
        
        try:
            # Capture current frame
            img_path = './detection_frame.jpg'
            cv2.imwrite(img_path, Vilib.img)
            
            # Run object detection
            gray_print("Scanning for objects...")
            result = vision_helper.detect_objects(img_path)
            last_detection_time = current_time
            
            if result.get('detected') and result.get('should_stop'):
                objects = result.get('objects', [])
                summary = result.get('summary', '')
                conversation = result.get('conversation_starter', '')
                
                if objects:
                    # Found something interesting!
                    my_car.stop()  # Briefly stop to announce
                    
                    # Build announcement
                    object_names = [obj['name'] for obj in objects]
                    object_types = [obj['type'] for obj in objects]
                    
                    # Check for dogs or cats specifically
                    has_animal = any(t in ['dog', 'cat'] for t in object_types)
                    
                    if has_animal:
                        led.on()  # Light up for animals!
                    
                    # Announce what we see
                    print(f"\n🔍 DETECTED: {', '.join(object_names)}")
                    print(f"   Summary: {summary}")
                    
                    state.last_detection = result
                    
                    # Speak the discovery
                    if summary:
                        speak(summary)
                    
                    # Brief comment then continue exploring
                    if conversation:
                        time.sleep(0.3)
                        speak("Interesting! Let me keep exploring.")
                    
                    led.off()
                    
                    # Resume roaming after announcement
                    print("🚗 Resuming exploration...")
                    time.sleep(0.5)
                    # state.roaming remains True, so car continues
                    
        except Exception as e:
            print(f"Detection loop error: {e}")
        
        time.sleep(0.2)

# ============================================================================
# Roaming Logic
# ============================================================================
def scan_for_clearance():
    """Scan left, center, and right to find distances. Returns (left, center, right) distances."""
    # Look left
    my_car.set_cam_pan_angle(60)
    time.sleep(0.3)
    left_distance = my_car.get_distance()
    if left_distance < 0:
        left_distance = 999  # No obstacle = clear
    
    # Look center
    my_car.set_cam_pan_angle(0)
    time.sleep(0.3)
    center_distance = my_car.get_distance()
    if center_distance < 0:
        center_distance = 999
    
    # Look right
    my_car.set_cam_pan_angle(-60)
    time.sleep(0.3)
    right_distance = my_car.get_distance()
    if right_distance < 0:
        right_distance = 999
    
    # Reset camera
    my_car.set_cam_pan_angle(0)
    
    return left_distance, center_distance, right_distance

def find_safe_path():
    """
    Scan surroundings and find the safest path.
    Returns turn angle (positive=left, negative=right) or None if stuck.
    """
    left_dist, center_dist, right_dist = scan_for_clearance()
    
    gray_print(f"Scan results - Left: {left_dist}cm, Center: {center_dist}cm, Right: {right_dist}cm")
    
    # Check if we have enough clearance (considering car width of ~45cm)
    left_clear = left_dist >= SAFE_CLEARANCE
    right_clear = right_dist >= SAFE_CLEARANCE
    center_clear = center_dist >= SAFE_CLEARANCE
    
    if center_clear:
        # Center is clear, go straight
        return 0
    elif left_clear and right_clear:
        # Both sides clear, pick the more open one
        if left_dist > right_dist:
            return 35  # Turn left
        else:
            return -35  # Turn right
    elif left_clear:
        return 35  # Turn left
    elif right_clear:
        return -35  # Turn right
    else:
        # Neither side has enough clearance - need to back up more
        return None

def backup_and_navigate():
    """Back up from obstacle and find a safe path around it."""
    gray_print("Backing up to get clearance...")
    
    # Back up to get some distance from the obstacle
    my_car.set_dir_servo_angle(0)
    my_car.backward(BACKUP_SPEED)
    time.sleep(0.8)
    my_car.stop()
    time.sleep(0.2)
    
    # Try to find a safe path
    for attempt in range(3):  # Try up to 3 times
        turn_angle = find_safe_path()
        
        if turn_angle is not None:
            gray_print(f"Found path! Turning {turn_angle} degrees")
            
            # Execute the turn
            my_car.set_dir_servo_angle(turn_angle)
            my_car.forward(ROAM_SPEED)
            time.sleep(1.0)  # Turn for a full second to get around obstacle
            my_car.set_dir_servo_angle(0)
            return True
        else:
            # Still stuck - back up more
            gray_print(f"Attempt {attempt + 1}: Still blocked, backing up more...")
            my_car.backward(BACKUP_SPEED)
            time.sleep(0.6)
            my_car.stop()
            time.sleep(0.2)
    
    # After 3 attempts, try a sharp turn as last resort
    gray_print("Path blocked - executing sharp turn to escape")
    turn_direction = random.choice([-1, 1])
    my_car.set_dir_servo_angle(35 * turn_direction)
    my_car.forward(ROAM_SPEED)
    time.sleep(1.5)
    my_car.set_dir_servo_angle(0)
    return True

def roam_step():
    """Execute one step of roaming behavior with obstacle avoidance."""
    if not state.roaming or state.speaking:
        return
    
    try:
        # Get ultrasonic distance
        distance = my_car.get_distance()
        
        # Negative values mean no object detected (out of range) - path is clear!
        # Ultrasonic sensors return -1 or -2 when nothing is in range (typically >400cm)
        if distance < 0:
            # No obstacle detected - safe to proceed
            if random.random() < 0.1:  # 10% chance to make a slight turn for exploration
                slight_turn = random.randint(-15, 15)
                my_car.set_dir_servo_angle(slight_turn)
            else:
                my_car.set_dir_servo_angle(0)
            my_car.forward(ROAM_SPEED)
            return
        
        if distance < CRITICAL_DISTANCE:
            # Critical! Stop immediately and back up
            gray_print(f"⚠️ Critical distance: {distance}cm - emergency backup!")
            my_car.stop()
            time.sleep(0.1)
            backup_and_navigate()
            
        elif distance < MIN_OBSTACLE_DISTANCE:
            # Obstacle ahead - stop, back up, and find a path around it
            gray_print(f"🚧 Obstacle detected at {distance}cm")
            my_car.stop()
            time.sleep(0.1)
            backup_and_navigate()
            
        else:
            # Path is clear - proceed forward with slight random variations
            if random.random() < 0.08:  # 8% chance to make a slight turn for exploration
                slight_turn = random.randint(-12, 12)
                my_car.set_dir_servo_angle(slight_turn)
            else:
                my_car.set_dir_servo_angle(0)
            
            my_car.forward(ROAM_SPEED)
            
    except Exception as e:
        print(f"Roaming error: {e}")
        my_car.stop()

def roaming_loop():
    """Main roaming control loop."""
    while True:
        if state.roaming and not state.speaking:
            roam_step()
        else:
            my_car.stop()
        
        time.sleep(0.1)

# ============================================================================
# Voice Command Loop
# ============================================================================
def voice_command_loop():
    """Listen for voice commands and handle conversations."""
    while True:
        if state.speaking:
            time.sleep(0.1)
            continue
        
        gray_print("Listening for commands...")
        
        command = listen_for_command(timeout=5)
        
        if command:
            print(f"Heard: {command}")
            
            # Check for wake words
            wake_action = check_wake_words(command)
            
            if wake_action == 'stop':
                state.roaming = False
                state.conversation_mode = False
                my_car.stop()
                print("\n🛑 STOPPING")
                speak("Stopping! I'll stay right here.")
                
            elif wake_action == 'start':
                if not state.roaming:
                    print("\n🚗 STARTING ROAM MODE")
                    speak("Let's go exploring! I'll look for interesting things.")
                    state.conversation_mode = False
                    time.sleep(0.5)
                    state.roaming = True
                    
            elif state.conversation_mode or not state.roaming:
                # In conversation mode or stopped - chat about what we see
                led.on()
                
                # Include context about last detection if available
                context = ""
                if state.last_detection:
                    context = f" (Context: I just saw {state.last_detection.get('summary', 'something interesting')})"
                
                # Get response from GPT
                img_path = './conversation_frame.jpg'
                cv2.imwrite(img_path, Vilib.img)
                
                response = openai_helper.dialogue_with_img(command + context, img_path)
                
                if isinstance(response, dict):
                    answer = response.get('answer', '')
                    actions = response.get('actions', [])
                    
                    # Perform any actions
                    for action in actions:
                        if action in actions_dict:
                            try:
                                actions_dict[action](my_car)
                            except Exception as e:
                                print(f"Action error: {e}")
                        elif action in sounds_dict:
                            try:
                                sounds_dict[action](music)
                            except Exception as e:
                                print(f"Sound error: {e}")
                    
                    if answer:
                        speak(answer)
                else:
                    if response:
                        speak(str(response))
                
                led.off()
        
        time.sleep(0.1)

# ============================================================================
# Keyboard Input Loop (Alternative to Voice)
# ============================================================================
def keyboard_loop():
    """Handle keyboard input for testing without voice."""
    import readline  # Better input handling
    
    print("\nKeyboard commands:")
    print("  'stop' - Stop the car")
    print("  'start' - Start roaming")
    print("  'scan' - Force object scan")
    print("  Any other text - Have a conversation")
    print()
    
    while True:
        try:
            user_input = input(f"\033[1;30m{'> '}\033[0m").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'stop':
                state.roaming = False
                my_car.stop()
                print("🛑 Stopped")
                speak("Stopping.")
                
            elif user_input.lower() == 'start':
                print("🚗 Starting roam mode")
                speak("Let's go exploring!")
                time.sleep(0.5)
                state.roaming = True
                
            elif user_input.lower() == 'scan':
                # Force an object scan
                img_path = './manual_scan.jpg'
                cv2.imwrite(img_path, Vilib.img)
                
                print("Scanning...")
                result = vision_helper.detect_objects(img_path)
                
                if result.get('detected'):
                    print(f"Detected: {result}")
                    summary = result.get('summary', '')
                    if summary:
                        speak(summary)
                else:
                    speak("I don't see anything particularly interesting right now.")
                    
            else:
                # Regular conversation
                led.on()
                
                img_path = './keyboard_frame.jpg'
                cv2.imwrite(img_path, Vilib.img)
                
                response = openai_helper.dialogue_with_img(user_input, img_path)
                
                if isinstance(response, dict):
                    answer = response.get('answer', '')
                    actions = response.get('actions', [])
                    
                    for action in actions:
                        if action in actions_dict:
                            try:
                                actions_dict[action](my_car)
                            except:
                                pass
                        elif action in sounds_dict:
                            try:
                                sounds_dict[action](music)
                            except:
                                pass
                    
                    if answer:
                        speak(answer)
                else:
                    if response:
                        speak(str(response))
                
                led.off()
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Input error: {e}")

# ============================================================================
# Main
# ============================================================================
def main():
    """Main entry point."""
    my_car.reset()
    my_car.set_cam_tilt_angle(20)
    my_car.set_cam_pan_angle(0)
    
    # Start threads
    detection_thread = threading.Thread(target=object_detection_loop, daemon=True)
    roaming_thread = threading.Thread(target=roaming_loop, daemon=True)
    
    detection_thread.start()
    roaming_thread.start()
    
    # Welcome message
    welcome_msg = f"Hello! I'm {ROBOT_NAME}, your roaming companion. Say 'Start' to begin exploring, or 'Stop' to pause. I'll look out for dogs, cats, and interesting objects!"
    print(f"\n{welcome_msg}\n")
    speak(welcome_msg)
    
    # Start appropriate input loop
    if input_mode == 'keyboard':
        keyboard_loop()
    else:
        voice_command_loop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\033[31mERROR: {e}\033[0m")
    finally:
        state.roaming = False
        my_car.stop()
        time.sleep(0.2)
        Vilib.camera_close()
        my_car.reset()
        print("Goodbye!")

