#!/usr/bin/env python3
"""
Security Patrol for PiCar-X - "Head of Security"

This script enables the PiCar-X to:
1. Patrol the room slowly as a security guard
2. Detect humans and animals using OpenAI's vision model
3. Send email alerts with photos when intruders are detected
4. Pause patrolling if no detections occur within a configurable timeout
5. Support voice wake words: "Stop" to stop, "Patrol" to resume

Usage:
    sudo python3 security_patrol.py
    sudo python3 security_patrol.py --keyboard
"""

from openai_helper import OpenAiHelper
from keys import (
    OPENAI_API_KEY, OPENAI_ASSISTANT_ID,
    SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD,
    EMAIL_FROM, EMAIL_TO
)
from preset_actions import actions_dict, sounds_dict
from utils import (
    gray_print, warn, error,
    redirect_error_2_null, cancel_redirect_error,
    sox_volume, speak_block
)

import speech_recognition as sr
from openai import OpenAI

from picarx import Picarx
from robot_hat import Music, Pin, ADC

import time
import threading
import random
import os
import sys
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime

os.popen("pinctrl set 20 op dh")  # enable robot_hat speaker switch
current_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_path)  # change working directory

# ============================================================================
# Configuration
# ============================================================================
ROBOT_NAME = "Scout"
SECURITY_TITLE = "Head of Security"

# Input mode
input_mode = 'voice'
args = sys.argv[1:]
if '--keyboard' in args:
    input_mode = 'keyboard'

# ============================================================================
# EMAIL CONFIGURATION
# ============================================================================
# Email credentials are imported from keys.py (SMTP_SERVER, SMTP_PORT, 
# SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO)

EMAIL_ENABLED = True  # Set to False to disable email alerts

# Email cooldown to prevent spam (seconds between emails for same detection type)
EMAIL_COOLDOWN = 60  # Don't send another email for same type within 60 seconds

# ============================================================================
# PATROL CONFIGURATION
# ============================================================================
# Patrol speed (slower than normal roaming for careful observation)
PATROL_SPEED = 20  # Forward speed (0-100) - slow and steady
TURN_SPEED = 25    # Turning speed
BACKUP_SPEED = 20  # Reverse speed

# Obstacle avoidance
MIN_OBSTACLE_DISTANCE = 40  # cm - stop/turn when obstacle is closer
CRITICAL_DISTANCE = 20      # cm - immediate stop and reverse
CAR_WIDTH = 45              # cm - car width for clearance calculations
SAFE_CLEARANCE = 55         # cm - clearance needed to navigate around obstacles

# Stuck detection parameters
STUCK_DISTANCE_THRESHOLD = 8
STUCK_CHECK_COUNT = 2
MAX_STUCK_ESCAPES = 3

# ============================================================================
# DETECTION & TIMEOUT CONFIGURATION
# ============================================================================
# Object detection parameters
DETECTION_INTERVAL = 3.0  # seconds between detection scans (slower for patrol)
TARGETS_OF_INTEREST = ['human', 'person', 'man', 'woman', 'child', 'dog', 'cat', 'animal']

# Idle timeout - pause patrolling if no detection within this time (seconds)
# Set to 0 to disable timeout (patrol indefinitely)
IDLE_TIMEOUT_SECONDS = 120  # 2 minutes (configurable)

# ============================================================================
# AUDIO SETTINGS
# ============================================================================
LANGUAGE = ['en']
VOLUME_DB = 3
TTS_VOICE = 'onyx'  # Deeper voice for security persona
VOICE_INSTRUCTIONS = "Speak with authority and professionalism, like a security guard. Be alert and observant."

# Wake words for control
WAKE_WORD_STOP = ['stop', 'freeze', 'halt', 'stand down']
WAKE_WORD_START = ['patrol', 'start', 'go', 'secure', 'guard']

# ============================================================================
# Email Handler
# ============================================================================
class EmailAlert:
    """Handles sending email alerts with images."""
    
    def __init__(self):
        self.last_email_times = {}  # Track last email time by detection type
    
    def can_send_email(self, detection_type):
        """Check if enough time has passed since last email for this type."""
        if detection_type not in self.last_email_times:
            return True
        
        elapsed = time.time() - self.last_email_times[detection_type]
        return elapsed >= EMAIL_COOLDOWN
    
    def send_alert(self, image_path, detection_type, description, objects_detected):
        """
        Send an email alert with the captured image.
        
        Args:
            image_path: Path to the captured image
            detection_type: 'human' or 'animal'
            description: Description of what was detected
            objects_detected: List of detected objects
        """
        if not EMAIL_ENABLED:
            print("📧 Email alerts disabled - would have sent alert")
            return False
        
        if not self.can_send_email(detection_type):
            gray_print(f"Email cooldown active for {detection_type} - skipping")
            return False
        
        # Record attempt time BEFORE trying to send to prevent spam on repeated failures
        self.last_email_times[detection_type] = time.time()
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = EMAIL_FROM
            msg['To'] = ', '.join(EMAIL_TO)
            msg['Subject'] = f"🚨 SECURITY ALERT: {detection_type.upper()} DETECTED - {ROBOT_NAME}"
            
            # Email body
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            body = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔒 SECURITY PATROL ALERT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 Robot: {ROBOT_NAME} ({SECURITY_TITLE})
📅 Time: {timestamp}
🎯 Detection Type: {detection_type.upper()}

📝 Description:
{description}

🔍 Objects Detected:
{chr(10).join(f'  • {obj}' for obj in objects_detected)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This is an automated alert from your security patrol robot.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach image
            if os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                    image = MIMEImage(img_data, _subtype='jpeg')
                    image.add_header('Content-Disposition', 'attachment', 
                                   filename=f"security_capture_{timestamp.replace(':', '-')}.jpg")
                    msg.attach(image)
            
            # Send email
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
            
            print(f"📧 Security alert email sent successfully!")
            return True
            
        except Exception as e:
            error(f"Failed to send email alert: {e}")
            return False

# ============================================================================
# Vision Helper for Security Detection
# ============================================================================
class SecurityVisionHelper:
    """Helper class for detecting humans and animals."""
    
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, timeout=30)
    
    def encode_image_to_base64(self, image_path):
        """Encode image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def detect_intruders(self, image_path):
        """
        Analyze image to detect humans and animals (security threats).
        Returns dict with detected intruders and descriptions.
        """
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a security surveillance system for a patrol robot.
                        Your job is to detect humans and animals in the camera feed.
                        
                        Respond in this exact JSON format:
                        {
                            "detected": true/false,
                            "detection_type": "human" or "animal" or "none",
                            "intruders": [
                                {"name": "description", "type": "human/animal", "threat_level": "low/medium/high"}
                            ],
                            "summary": "A security-style description of what you observe",
                            "alert_required": true/false,
                            "security_note": "Brief professional security observation"
                        }
                        
                        Set "alert_required" to true if you detect:
                        - Any human (person, man, woman, child)
                        - Any animal (dog, cat, bird, etc.)
                        
                        threat_level guidelines:
                        - low: Person appears to belong there, pets
                        - medium: Unknown person, larger animals
                        - high: Multiple people, suspicious behavior
                        
                        Be thorough but avoid false positives. Only report actual humans/animals."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Security scan: Analyze this image for any humans or animals. Report all detected subjects."
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
                "detection_type": "none",
                "intruders": [],
                "summary": result_text,
                "alert_required": False,
                "security_note": ""
            }
            
        except Exception as e:
            print(f"Vision detection error: {e}")
            return {
                "detected": False,
                "detection_type": "none",
                "intruders": [],
                "summary": "",
                "alert_required": False,
                "security_note": ""
            }

# ============================================================================
# Initialize Components
# ============================================================================
print(f"\n{'='*60}")
print(f"  {ROBOT_NAME} - {SECURITY_TITLE}")
print(f"  Security Patrol Mode")
print(f"  Say 'Stop' to halt, 'Patrol' to resume")
print(f"  Idle timeout: {IDLE_TIMEOUT_SECONDS}s ({IDLE_TIMEOUT_SECONDS/60:.1f} min)")
print(f"{'='*60}\n")

# Initialize helpers
openai_helper = OpenAiHelper(OPENAI_API_KEY, OPENAI_ASSISTANT_ID, 'picarx')
vision_helper = SecurityVisionHelper(OPENAI_API_KEY)
email_alert = EmailAlert()

# Car initialization
try:
    my_car = Picarx()
    time.sleep(1)
except Exception as e:
    raise RuntimeError(f"Failed to initialize PiCar-X: {e}")

music = Music()
led = Pin('LED')

# Battery monitoring (ADC pin A4 for battery voltage)
battery_adc = ADC("A4")

def get_battery_info():
    """
    Read battery voltage and estimate percentage.
    PiCar-X uses 2S Li-ion battery (7.4V nominal, 8.4V full, 6.4V empty).
    The ADC reads through a voltage divider (3:1 ratio).
    """
    try:
        # Read ADC value (0-4095 for 12-bit ADC, maps to 0-3.3V)
        adc_value = battery_adc.read()
        # Convert to actual voltage (3.3V reference, 3:1 voltage divider)
        voltage = (adc_value / 4095.0) * 3.3 * 3
        
        # Estimate percentage based on 2S Li-ion discharge curve
        # Full: 8.4V (100%), Nominal: 7.4V (50%), Empty: 6.4V (0%)
        if voltage >= 8.4:
            percentage = 100
        elif voltage <= 6.4:
            percentage = 0
        else:
            # Linear approximation between 6.4V and 8.4V
            percentage = int((voltage - 6.4) / (8.4 - 6.4) * 100)
        
        return voltage, percentage
    except Exception as e:
        print(f"Battery read error: {e}")
        return None, None

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
class PatrolState:
    """Thread-safe state management for the security patrol robot."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.car_lock = threading.Lock()  # Lock for car control synchronization
        self._patrolling = False
        self._speaking = False
        self._paused_due_to_timeout = False
        self._handling_detection = False  # Flag to indicate detection in progress
        self._last_detection_time = time.time()  # Track last detection
        self._total_detections = 0
        self._last_distance = -1
        self._stuck_counter = 0
        self._last_progress_time = time.time()
        self._escape_counter = 0
        self._last_escape_time = 0
    
    @property
    def handling_detection(self):
        with self.lock:
            return self._handling_detection
    
    @handling_detection.setter
    def handling_detection(self, value):
        with self.lock:
            self._handling_detection = value
    
    @property
    def patrolling(self):
        with self.lock:
            return self._patrolling
    
    @patrolling.setter
    def patrolling(self, value):
        with self.lock:
            self._patrolling = value
    
    @property
    def speaking(self):
        with self.lock:
            return self._speaking
    
    @speaking.setter
    def speaking(self, value):
        with self.lock:
            self._speaking = value
    
    @property
    def paused_due_to_timeout(self):
        with self.lock:
            return self._paused_due_to_timeout
    
    @paused_due_to_timeout.setter
    def paused_due_to_timeout(self, value):
        with self.lock:
            self._paused_due_to_timeout = value
    
    @property
    def last_detection_time(self):
        with self.lock:
            return self._last_detection_time
    
    @last_detection_time.setter
    def last_detection_time(self, value):
        with self.lock:
            self._last_detection_time = value
    
    @property
    def total_detections(self):
        with self.lock:
            return self._total_detections
    
    def increment_detections(self):
        with self.lock:
            self._total_detections += 1
            self._last_detection_time = time.time()
    
    @property
    def stuck_counter(self):
        with self.lock:
            return self._stuck_counter
    
    @stuck_counter.setter
    def stuck_counter(self, value):
        with self.lock:
            self._stuck_counter = value
    
    @property
    def last_distance(self):
        with self.lock:
            return self._last_distance
    
    @last_distance.setter
    def last_distance(self, value):
        with self.lock:
            self._last_distance = value
    
    @property
    def last_progress_time(self):
        with self.lock:
            return self._last_progress_time
    
    @last_progress_time.setter
    def last_progress_time(self, value):
        with self.lock:
            self._last_progress_time = value
    
    @property
    def escape_counter(self):
        with self.lock:
            return self._escape_counter
    
    @escape_counter.setter
    def escape_counter(self, value):
        with self.lock:
            self._escape_counter = value
    
    @property
    def last_escape_time(self):
        with self.lock:
            return self._last_escape_time
    
    @last_escape_time.setter
    def last_escape_time(self, value):
        with self.lock:
            self._last_escape_time = value
    
    def reset_stuck_detection(self):
        """Thread-safe method to reset all stuck detection state."""
        with self.lock:
            self._stuck_counter = 0
            self._last_distance = -1
            self._escape_counter = 0
            self._last_progress_time = time.time()
            self._last_escape_time = 0
    
    def reset_detection_timer(self):
        """Reset the detection timer (called when starting patrol)."""
        with self.lock:
            self._last_detection_time = time.time()
            self._paused_due_to_timeout = False

state = PatrolState()

# ============================================================================
# TTS Handler
# ============================================================================
def speak(text, blocking=True):
    """Convert text to speech and play it."""
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
            if sox_volume(_tts_raw, _tts_final, VOLUME_DB):
                speak_block(music, _tts_final)
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
            _stderr_back = None
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            try:
                audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
            except sr.WaitTimeoutError:
                return None
        
        result = openai_helper.stt(audio, language=LANGUAGE[0])
        return result if result else None
        
    except Exception as e:
        return None
    finally:
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
# Security Detection Thread
# ============================================================================
def security_detection_loop():
    """Continuously monitor camera for humans and animals."""
    last_detection_time = 0
    
    while True:
        if not state.patrolling or state.speaking:
            time.sleep(0.5)
            continue
        
        current_time = time.time()
        if current_time - last_detection_time < DETECTION_INTERVAL:
            time.sleep(0.2)
            continue
        
        try:
            # Capture current frame
            img_path = './security_frame.jpg'
            cv2.imwrite(img_path, Vilib.img)
            
            # Run security detection
            gray_print("🔍 Security scan in progress...")
            result = vision_helper.detect_intruders(img_path)
            last_detection_time = current_time
            
            if result.get('detected') and result.get('alert_required'):
                intruders = result.get('intruders', [])
                summary = result.get('summary', '')
                detection_type = result.get('detection_type', 'unknown')
                security_note = result.get('security_note', '')
                
                if intruders:
                    # INTRUDER DETECTED! Acquire car lock to prevent race with patrol loop
                    speech_text = None
                    
                    try:
                        # Set flag inside try block so finally always resets it
                        state.handling_detection = True
                        
                        # Phase 1: Car control operations (with lock)
                        with state.car_lock:
                            my_car.stop()  # Stop to capture clear image
                            led.on()  # Alert light
                        
                        # Update detection stats
                        state.increment_detections()
                        
                        # Get intruder names for logging
                        intruder_names = [i.get('name', 'Unknown') for i in intruders]
                        threat_levels = [i.get('threat_level', 'unknown') for i in intruders]
                        
                        print(f"\n🚨 SECURITY ALERT: {detection_type.upper()} DETECTED!")
                        print(f"   Subjects: {', '.join(intruder_names)}")
                        print(f"   Threat levels: {', '.join(threat_levels)}")
                        print(f"   Summary: {summary}")
                        
                        # Capture high-quality image for email
                        alert_img_path = f'./security_alerts/alert_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
                        os.makedirs('./security_alerts', exist_ok=True)
                        cv2.imwrite(alert_img_path, Vilib.img)
                        
                        # Send email alert
                        email_alert.send_alert(
                            alert_img_path,
                            detection_type,
                            summary,
                            intruder_names
                        )
                        
                        # Prepare speech text (will be spoken outside lock)
                        if summary:
                            speech_text = f"Security alert! {summary}"
                        
                        led.off()
                        
                        # Phase 2: Announce detection (without lock - allows patrol to check state)
                        if speech_text:
                            speak(speech_text)
                        
                        # Brief pause then continue patrol
                        time.sleep(1.0)
                        
                        # Phase 3: Check path clearance (with lock for car control)
                        with state.car_lock:
                            distance = my_car.get_distance()
                            if distance >= 0 and distance < SAFE_CLEARANCE:
                                backup_and_navigate()
                                state.stuck_counter = 0
                                state.last_distance = -1
                    finally:
                        state.handling_detection = False
                    
        except Exception as e:
            print(f"Detection loop error: {e}")
        
        time.sleep(0.2)

# ============================================================================
# Idle Timeout Monitor
# ============================================================================
def idle_timeout_loop():
    """Monitor for idle timeout and pause patrol if no detections."""
    if IDLE_TIMEOUT_SECONDS <= 0:
        return  # Timeout disabled
    
    while True:
        if state.patrolling and not state.paused_due_to_timeout:
            time_since_detection = time.time() - state.last_detection_time
            
            if time_since_detection >= IDLE_TIMEOUT_SECONDS:
                # Timeout reached - pause patrol
                print(f"\n⏸️ IDLE TIMEOUT: No detections for {IDLE_TIMEOUT_SECONDS}s")
                print(f"   Pausing patrol to conserve energy...")
                
                state.patrolling = False
                state.paused_due_to_timeout = True
                
                # Acquire car lock to safely stop (prevent race with patrol/detection loops)
                with state.car_lock:
                    my_car.stop()
                
                speak(f"No activity detected for {int(IDLE_TIMEOUT_SECONDS/60)} minutes. "
                      f"Entering standby mode. Say 'Patrol' to resume.")
        
        time.sleep(5)  # Check every 5 seconds

# ============================================================================
# Patrol Navigation Logic (adapted from roaming)
# ============================================================================
def check_if_stuck(current_distance):
    """Check if the car is stuck."""
    current_time = time.time()
    
    if current_distance < 0:
        if state.last_distance >= 0:
            state.stuck_counter = 0
            state.last_progress_time = current_time
        else:
            time_in_open = current_time - state.last_progress_time
            if time_in_open > 7.0:
                state.last_progress_time = current_time
                state.last_distance = current_distance
                return True
        
        state.last_distance = current_distance
        return False
    
    time_since_progress = current_time - state.last_progress_time
    if time_since_progress > 5.0 and state.last_distance >= 0:
        avg_distance_change = abs(current_distance - state.last_distance)
        if avg_distance_change < 20:
            state.last_progress_time = current_time
            return True
    
    if state.last_distance >= 0:
        distance_change = abs(current_distance - state.last_distance)
        
        if distance_change < STUCK_DISTANCE_THRESHOLD:
            state.stuck_counter += 1
            
            if state.stuck_counter >= STUCK_CHECK_COUNT:
                state.last_progress_time = current_time
                return True
        else:
            state.last_progress_time = current_time
            state.stuck_counter = 0
    elif state.last_distance < 0:
        state.stuck_counter = 0
        state.last_progress_time = current_time
    else:
        state.last_progress_time = current_time
    
    state.last_distance = current_distance
    return False

def aggressive_escape():
    """Perform an aggressive escape maneuver when stuck."""
    current_time = time.time()
    
    if current_time - state.last_escape_time < 10:
        state.escape_counter += 1
    else:
        state.escape_counter = 1
    
    state.last_escape_time = current_time
    state.stuck_counter = 0
    
    my_car.stop()
    time.sleep(0.2)
    
    if state.escape_counter >= MAX_STUCK_ESCAPES:
        gray_print("⚠️ Repeatedly stuck - executing 180° turn")
        state.escape_counter = 0
        
        my_car.backward(BACKUP_SPEED + 10)
        time.sleep(1.5)
        my_car.stop()
        time.sleep(0.2)
        
        turn_direction = random.choice([-1, 1])
        my_car.set_dir_servo_angle(40 * turn_direction)
        my_car.forward(TURN_SPEED)
        time.sleep(3.0)
        my_car.set_dir_servo_angle(0)
        my_car.stop()
        time.sleep(0.2)
        
        my_car.forward(PATROL_SPEED)
        time.sleep(1.0)
        my_car.stop()
        
        state.last_distance = -1
        return
    
    # Normal escape: wiggle backward
    for i in range(4):
        turn = 35 if i % 2 == 0 else -35
        my_car.set_dir_servo_angle(turn)
        my_car.backward(BACKUP_SPEED + 10)
        time.sleep(0.5)
    
    my_car.stop()
    my_car.set_dir_servo_angle(0)
    time.sleep(0.3)
    
    left_dist, center_dist, right_dist = scan_for_clearance()
    
    if left_dist > right_dist:
        escape_angle = 40
    else:
        escape_angle = -40
    
    my_car.set_dir_servo_angle(escape_angle)
    my_car.forward(PATROL_SPEED)
    time.sleep(2.0)
    my_car.set_dir_servo_angle(0)
    my_car.stop()
    time.sleep(0.2)
    
    state.last_distance = -1
    state.stuck_counter = 0

def scan_for_clearance():
    """Scan left, center, and right to find distances."""
    my_car.set_cam_pan_angle(60)
    time.sleep(0.3)
    left_distance = my_car.get_distance()
    if left_distance < 0:
        left_distance = 999
    
    my_car.set_cam_pan_angle(0)
    time.sleep(0.3)
    center_distance = my_car.get_distance()
    if center_distance < 0:
        center_distance = 999
    
    my_car.set_cam_pan_angle(-60)
    time.sleep(0.3)
    right_distance = my_car.get_distance()
    if right_distance < 0:
        right_distance = 999
    
    my_car.set_cam_pan_angle(0)
    
    return left_distance, center_distance, right_distance

def initial_clearance_check():
    """Check path clearance before starting patrol."""
    print("🔍 Checking patrol route clearance...")
    
    left_dist, center_dist, right_dist = scan_for_clearance()
    print(f"   Left: {left_dist:.0f}cm | Center: {center_dist:.0f}cm | Right: {right_dist:.0f}cm")
    
    if center_dist >= SAFE_CLEARANCE:
        print("✅ Forward path is clear!")
        return True
    
    print(f"⚠️ Forward path blocked ({center_dist:.0f}cm) - finding clear route...")
    
    if left_dist >= SAFE_CLEARANCE or right_dist >= SAFE_CLEARANCE:
        if left_dist > right_dist:
            turn_angle = 35
            direction = "LEFT"
        else:
            turn_angle = -35
            direction = "RIGHT"
        
        print(f"🔄 Turning {direction}")
        
        my_car.set_dir_servo_angle(turn_angle)
        my_car.forward(PATROL_SPEED)
        time.sleep(1.5)
        my_car.set_dir_servo_angle(0)
        my_car.stop()
        time.sleep(0.2)
        
        return True
    
    print("🚧 All directions blocked - backing up...")
    my_car.backward(BACKUP_SPEED)
    time.sleep(1.0)
    my_car.stop()
    time.sleep(0.2)
    
    left_dist, center_dist, right_dist = scan_for_clearance()
    max_dist = max(left_dist, center_dist, right_dist)
    
    if max_dist >= SAFE_CLEARANCE:
        if max_dist == center_dist:
            turn_angle = 0
        elif max_dist == left_dist:
            turn_angle = 35
        else:
            turn_angle = -35
        
        if turn_angle != 0:
            my_car.set_dir_servo_angle(turn_angle)
            my_car.forward(PATROL_SPEED)
            time.sleep(1.5)
            my_car.set_dir_servo_angle(0)
            my_car.stop()
        
        return True
    
    print("❌ Could not find clear path")
    return False

def find_safe_path():
    """Scan surroundings and find the safest path."""
    left_dist, center_dist, right_dist = scan_for_clearance()
    
    left_clear = left_dist >= SAFE_CLEARANCE
    right_clear = right_dist >= SAFE_CLEARANCE
    center_clear = center_dist >= SAFE_CLEARANCE
    
    if center_clear:
        return 0
    elif left_clear and right_clear:
        return 35 if left_dist > right_dist else -35
    elif left_clear:
        return 35
    elif right_clear:
        return -35
    else:
        return None

def backup_and_navigate():
    """Back up from obstacle and find a safe path around it."""
    gray_print("Backing up to get clearance...")
    
    my_car.set_dir_servo_angle(0)
    my_car.backward(BACKUP_SPEED)
    time.sleep(0.8)
    my_car.stop()
    time.sleep(0.2)
    
    for attempt in range(3):
        turn_angle = find_safe_path()
        
        if turn_angle is not None:
            my_car.set_dir_servo_angle(turn_angle)
            my_car.forward(PATROL_SPEED)
            time.sleep(1.0)
            my_car.set_dir_servo_angle(0)
            my_car.stop()
            return True
        else:
            my_car.backward(BACKUP_SPEED)
            time.sleep(0.6)
            my_car.stop()
            time.sleep(0.2)
    
    turn_direction = random.choice([-1, 1])
    my_car.set_dir_servo_angle(35 * turn_direction)
    my_car.forward(PATROL_SPEED)
    time.sleep(1.5)
    my_car.set_dir_servo_angle(0)
    my_car.stop()
    return True

def patrol_step():
    """Execute one step of patrol behavior with obstacle avoidance."""
    # Skip if not patrolling, speaking, or detection handler has car control
    if not state.patrolling or state.speaking or state.handling_detection:
        return
    
    # Try to acquire car lock (non-blocking to avoid deadlock)
    if not state.car_lock.acquire(blocking=False):
        return  # Detection loop has control, skip this step
    
    try:
        distance = my_car.get_distance()
        
        if check_if_stuck(distance):
            aggressive_escape()
            return
        
        if distance < 0:
            # No obstacle - patrol forward with occasional slight turns
            if random.random() < 0.05:  # 5% chance for slight turn (less than roaming)
                slight_turn = random.randint(-10, 10)
                my_car.set_dir_servo_angle(slight_turn)
            else:
                my_car.set_dir_servo_angle(0)
            my_car.forward(PATROL_SPEED)
            return
        
        if distance < CRITICAL_DISTANCE:
            gray_print(f"⚠️ Critical distance: {distance}cm")
            my_car.stop()
            time.sleep(0.1)
            backup_and_navigate()
            state.stuck_counter = 0
            state.last_distance = -1
            
        elif distance < MIN_OBSTACLE_DISTANCE:
            gray_print(f"🚧 Obstacle at {distance}cm")
            my_car.stop()
            time.sleep(0.1)
            backup_and_navigate()
            state.stuck_counter = 0
            state.last_distance = -1
            
        else:
            if random.random() < 0.05:
                slight_turn = random.randint(-8, 8)
                my_car.set_dir_servo_angle(slight_turn)
            else:
                my_car.set_dir_servo_angle(0)
            
            my_car.forward(PATROL_SPEED)
            
    except Exception as e:
        print(f"Patrol error: {e}")
        my_car.stop()
    finally:
        state.car_lock.release()

def patrol_loop():
    """Main patrol control loop."""
    while True:
        # Skip if detection handler has car control
        if state.patrolling and not state.speaking and not state.handling_detection:
            patrol_step()
        elif not state.handling_detection:
            # Only stop if detection loop doesn't have control
            with state.car_lock:
                my_car.stop()
        
        time.sleep(0.15)  # Slightly slower loop for patrol

# ============================================================================
# Voice Command Loop
# ============================================================================
def voice_command_loop():
    """Listen for voice commands."""
    while True:
        if state.speaking:
            time.sleep(0.1)
            continue
        
        gray_print("Listening for commands...")
        
        command = listen_for_command(timeout=5)
        
        if command:
            print(f"Heard: {command}")
            
            wake_action = check_wake_words(command)
            
            if wake_action == 'stop':
                state.patrolling = False
                state.paused_due_to_timeout = False
                my_car.stop()
                print("\n🛑 PATROL HALTED")
                speak("Security patrol halted. Standing by.")
                
            elif wake_action == 'start':
                if not state.patrolling:
                    print("\n🔒 STARTING SECURITY PATROL")
                    speak("Initiating security patrol. Scanning area.")
                    time.sleep(0.3)
                    
                    if initial_clearance_check():
                        speak("Area clear. Beginning patrol.")
                        state.reset_stuck_detection()
                        state.reset_detection_timer()
                        state.patrolling = True
                    else:
                        speak("Unable to establish patrol route. Please reposition.")
        
        time.sleep(0.1)

# ============================================================================
# Manual Drive Mode (Joystick with Arrow Keys)
# ============================================================================
MANUAL_DRIVE_SPEED = 30  # Speed for manual control

def get_key_nonblocking():
    """Get a keypress without blocking. Returns None if no key pressed."""
    import sys
    import tty
    import termios
    import select
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        # Check if input is available (timeout 0.05 seconds)
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist:
            ch = sys.stdin.read(1)
            # Check for escape sequence (arrow keys send: ESC [ A/B/C/D)
            if ch == '\x1b':
                # Wait a bit longer for the rest of the escape sequence
                rlist2, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist2:
                    ch2 = sys.stdin.read(1)
                    if ch2 == '[':
                        # Read the direction character
                        rlist3, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if rlist3:
                            ch3 = sys.stdin.read(1)
                            return '\x1b[' + ch3
                        # No third char, return what we have
                        return '\x1b['
                    # Not a bracket, return escape + char
                    return '\x1b' + ch2
                # Just escape key
                return ch
            return ch
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def manual_drive_mode():
    """Enter manual drive mode with arrow key controls."""
    print("\n🎮 MANUAL DRIVE MODE")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  ↑  Forward")
    print("  ↓  Backward")
    print("  ←  Turn Left")
    print("  →  Turn Right")
    print("  SPACE  Stop")
    print("  Q  Exit manual mode")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Press arrow keys to drive. Car keeps moving until SPACE.\n")
    
    # Stop any automatic patrol
    was_patrolling = state.patrolling
    state.patrolling = False
    my_car.stop()
    my_car.set_dir_servo_angle(0)
    
    # Drive state
    drive_direction = None  # None, 'forward', 'backward'
    steering_angle = 0      # -30 (left) to 30 (right)
    last_status = ""
    
    try:
        while True:
            # Check for key input (non-blocking)
            key = get_key_nonblocking()
            
            if key:
                if key == 'q' or key == 'Q':
                    print("\n🔒 Exiting manual drive mode")
                    my_car.stop()
                    my_car.set_dir_servo_angle(0)
                    break
                
                elif key == '\x1b[A':  # Up arrow - Forward
                    drive_direction = 'forward'
                
                elif key == '\x1b[B':  # Down arrow - Backward
                    drive_direction = 'backward'
                
                elif key == '\x1b[D':  # Left arrow - Turn Left
                    steering_angle = -30
                
                elif key == '\x1b[C':  # Right arrow - Turn Right
                    steering_angle = 30
                
                elif key == ' ':  # Space - Stop
                    drive_direction = None
                    steering_angle = 0
                
                elif key == '\x03':  # Ctrl+C
                    raise KeyboardInterrupt
            
            # Apply current drive state continuously
            my_car.set_dir_servo_angle(steering_angle)
            
            if drive_direction == 'forward':
                my_car.forward(MANUAL_DRIVE_SPEED)
            elif drive_direction == 'backward':
                my_car.backward(MANUAL_DRIVE_SPEED)
            else:
                my_car.stop()
            
            # Update status display
            if drive_direction == 'forward':
                if steering_angle < 0:
                    status = "↖️  Forward + Left "
                elif steering_angle > 0:
                    status = "↗️  Forward + Right"
                else:
                    status = "⬆️  Forward        "
            elif drive_direction == 'backward':
                if steering_angle < 0:
                    status = "↙️  Backward + Left "
                elif steering_angle > 0:
                    status = "↘️  Backward + Right"
                else:
                    status = "⬇️  Backward        "
            else:
                if steering_angle != 0:
                    status = "⏹️  Stopped (turning)"
                else:
                    status = "⏹️  Stopped          "
            
            if status != last_status:
                print(status, end='\r', flush=True)
                last_status = status
                
    except KeyboardInterrupt:
        print("\n🔒 Manual drive interrupted")
    finally:
        my_car.stop()
        my_car.set_dir_servo_angle(0)
        if was_patrolling:
            print("ℹ️  Type 'patrol' to resume automatic patrol")

# ============================================================================
# Keyboard Input Loop
# ============================================================================
def keyboard_loop():
    """Handle keyboard input for testing."""
    import readline
    
    print("\nSecurity Patrol Commands:")
    print("  'stop'   - Halt patrol")
    print("  'patrol' - Start/resume patrol")
    print("  'drive'  - Manual drive mode (arrow keys)")
    print("  'scan'   - Force security scan")
    print("  'status' - Show patrol status")
    print("  'test'   - Test email (requires valid config)")
    print()
    
    while True:
        try:
            user_input = input(f"\033[1;30m{'🔒 > '}\033[0m").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'stop':
                state.patrolling = False
                state.paused_due_to_timeout = False
                my_car.stop()
                print("🛑 Patrol halted")
                speak("Security patrol halted.")
                
            elif user_input.lower() in ['patrol', 'start']:
                if not state.patrolling:
                    print("🔒 Starting security patrol")
                    speak("Initiating security patrol.")
                    time.sleep(0.3)
                    
                    if initial_clearance_check():
                        speak("Beginning patrol.")
                        state.reset_stuck_detection()
                        state.reset_detection_timer()
                        state.patrolling = True
                    else:
                        speak("Unable to establish patrol route.")
                else:
                    print("ℹ️ Already patrolling")
                    speak("Security patrol already active.")
            
            elif user_input.lower() == 'drive':
                manual_drive_mode()
                
            elif user_input.lower() == 'scan':
                img_path = './manual_security_scan.jpg'
                cv2.imwrite(img_path, Vilib.img)
                
                print("🔍 Running security scan...")
                result = vision_helper.detect_intruders(img_path)
                
                if result.get('detected'):
                    print(f"🚨 Detection: {result}")
                    summary = result.get('summary', '')
                    if summary:
                        speak(f"Security scan complete. {summary}")
                else:
                    speak("Security scan complete. No threats detected.")
                    
            elif user_input.lower() == 'status':
                print(f"\n📊 PATROL STATUS")
                print(f"   Patrolling: {state.patrolling}")
                print(f"   Paused (timeout): {state.paused_due_to_timeout}")
                print(f"   Total detections: {state.total_detections}")
                time_since = time.time() - state.last_detection_time
                print(f"   Time since last detection: {time_since:.0f}s")
                print(f"   Idle timeout: {IDLE_TIMEOUT_SECONDS}s")
                print(f"   Email enabled: {EMAIL_ENABLED}")
                # Battery status
                voltage, percentage = get_battery_info()
                if voltage is not None:
                    battery_icon = "🔋" if percentage > 20 else "🪫"
                    print(f"   {battery_icon} Battery: {percentage}% ({voltage:.2f}V)")
                else:
                    print(f"   🔋 Battery: Unable to read")
                print()
                
            elif user_input.lower() == 'test':
                print("📧 Testing email configuration...")
                img_path = './test_email.jpg'
                cv2.imwrite(img_path, Vilib.img)
                
                success = email_alert.send_alert(
                    img_path,
                    "test",
                    "This is a test alert from the security patrol system.",
                    ["Test Subject"]
                )
                
                if success:
                    print("✅ Test email sent successfully!")
                else:
                    print("❌ Failed to send test email. Check your configuration.")
                    
            else:
                # Regular conversation
                led.on()
                
                img_path = './keyboard_frame.jpg'
                cv2.imwrite(img_path, Vilib.img)
                
                response = openai_helper.dialogue_with_img(
                    f"[Security Mode] {user_input}", 
                    img_path
                )
                
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
    detection_thread = threading.Thread(target=security_detection_loop, daemon=True)
    patrol_thread = threading.Thread(target=patrol_loop, daemon=True)
    timeout_thread = threading.Thread(target=idle_timeout_loop, daemon=True)
    
    detection_thread.start()
    patrol_thread.start()
    timeout_thread.start()
    
    # Welcome message
    welcome_msg = (
        f"Security system online. I'm {ROBOT_NAME}, your {SECURITY_TITLE}. "
        f"Say 'Patrol' to begin security sweep, or 'Stop' to halt. "
        f"I'll alert you if I detect any humans or animals."
    )
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
        print("\n\nShutting down security patrol...")
    except Exception as e:
        print(f"\033[31mERROR: {e}\033[0m")
    finally:
        state.patrolling = False
        my_car.stop()
        time.sleep(0.2)
        Vilib.camera_close()
        my_car.reset()
        print("Security patrol offline. Goodbye!")

