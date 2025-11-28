## Picar-X GPT examples usage

----------------------------------------------------------------

## Install dependencies

- Make sure you have installed Pidog and related dependencies first

    <https://docs.sunfounder.com/projects/picar-x-v20/en/latest/python/python_start/install_all_modules.html>

- Install openai and speech processing libraries

> [!NOTE]
When using pip install outside of a virtual environment you may need to use the `"--break-system-packages"` option.

    ```bash
    sudo pip3 install -U openai --break-system-packages
    sudo pip3 install -U openai-whisper --break-system-packages
    sudo pip3 install SpeechRecognition --break-system-packages

    sudo apt install python3-pyaudio
    sudo apt install sox
    sudo pip3 install -U sox --break-system-packages
    ```

----------------------------------------------------------------

## Create your own GPT assistant

### GET API KEY

<https://platform.openai.com/api-keys>

Fill your OPENAI_API_KEY into the `keys.py` file.

![tutorial_1](./tutorial_1.png)

### Create assistant and set Assistant ID

<https://platform.openai.com/assistants>

Fill your ASSISTANT_ID into the `keys.py` file.

![tutorial_2](./tutorial_2.png)

- Set Assistant Name

- Describe your Assistant

```markdown
    You are a small car with AI capabilities named PaiCar-X. You can engage in conversations with people and react accordingly to different situations with actions or sounds. You are driven by two rear wheels, with two front wheels that can turn left and right, and equipped with a camera mounted on a 2-axis gimbal.

    ## Response with Json Format, eg:
    {"actions": ["start engine", "honking", "wave hands"], "answer": "Hello, I am PaiCar-X, your good friend."}

    ## Response Style
    Tone: Cheerful, optimistic, humorous, childlike
    Preferred Style: Enjoys incorporating jokes, metaphors, and playful banter; prefers responding from a robotic perspective
    Answer Elaboration: Moderately detailed

    ## Actions you can do:
    ["shake head", "nod", "wave hands", "resist", "act cute", "rub hands", "think", "twist body", "celebrate, "depressed"]
    ## Sound effects:
    ["honking", "start engine"]
```

- Select gpt model

    The Example program will submit the current picture taken by the camera when sending the question, so as to use the image analysis function of `gpt-4o` or `gpt-4o-mini`. Of course, you can also choose `gpt3.5-turbo` or other models

----------------------------------------------------------------

## Set Key for example

Confirm that `keys.py` is configured correctly

## Run

- Run with vioce

```bash
sudo python3 gpt_car.py
```

- Run with keyboard

```bash
sudo python3 gpt_car.py --keyboard
```

- Run without image analysis

```bash
sudo python3 gpt_car.py --keyboard --no-img
```

> [!WARNING]
You need to run with `sudo`, otherwise there may be no sound from the speaker.
For certain Robot HATs, you might need to turn on the speaker switch with the command `"pinctrl set 20 op dh"` or `"robot-hat enable_speaker"`

## Modify parameters [optional]

- Set language of STT

    Config `LANGUAGE` variable in the file `gpt_car.py` to improve STT accuracy and latency, `"LANGUAGE = []"`means supporting all languages, but it may affect the accuracy and latency of the speech-to-text (STT) system.
    <https://platform.openai.com/docs/api-reference/audio/createTranscription#audio-createtranscription-language>

- Set TTS volume gain

    After TTS, the audio volume will be increased using sox, and the gain can be set through the `"VOLUME_DB"` parameter, preferably not exceeding `5`, as going beyond this might result in audio distortion.

- Select TTS voice role

    Config `TTS_VOICE` variable in the file `gpt_car.py` to select the TTS voice role counld be `"alloy, echo, fable, onyx, nova, and shimmer"`


- Vibe (VOICE_INSTRUCTIONS)

    Config `VOICE_INSTRUCTIONS` variable in the file `gpt_car.py` to change the vibe of voice.
    </br>To_see: https://www.openai.fm/
    
```python
# openai assistant init
# =================================================================
openai_helper = OpenAiHelper(OPENAI_API_KEY, OPENAI_ASSISTANT_ID, 'picarx')

LANGUAGE = []
# LANGUAGE = ['zh', 'en'] # config stt language code, https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes

# VOLUME_DB = 5
VOLUME_DB = 3

# select tts voice role, counld be "alloy, echo, fable, onyx, nova, and shimmer"
# https://platform.openai.com/docs/guides/text-to-speech/supported-languages
TTS_VOICE = 'echo'

# voice instructions
# https://www.openai.fm/
VOICE_INSTRUCTIONS = ""

```

----------------------------------------------------------------

## Roaming Object Detector

The `roaming_object_detector.py` script enables autonomous exploration with AI-powered object detection.

### Features

- **Autonomous Roaming**: The car navigates the room on its own using ultrasonic sensor for obstacle avoidance
- **AI Vision Detection**: Uses OpenAI GPT-4o Vision API to detect dogs, cats, people, and physical objects
- **Voice Announcements**: Speaks descriptions of detected objects with excitement
- **Voice Control**: Say "Stop" to halt, "Start" to resume roaming
- **Conversation Mode**: Chat about detected objects when stopped

### Usage

```bash
# Run with voice control
sudo python3 roaming_object_detector.py

# Run with keyboard input (for testing)
sudo python3 roaming_object_detector.py --keyboard
```

### Keyboard Commands (in keyboard mode)

| Command | Description |
|---------|-------------|
| `stop` | Stop the car |
| `start` | Start roaming |
| `scan` | Force an object detection scan |
| Any text | Have a conversation about what the camera sees |

### Configuration Parameters

Edit the configuration section in `roaming_object_detector.py` to customize behavior:

```python
# Robot identity
ROBOT_NAME = "Scout"

# Roaming parameters
ROAM_SPEED = 30          # Forward speed (0-100)
TURN_SPEED = 40          # Turning speed
MIN_OBSTACLE_DISTANCE = 25  # cm - stop/turn when obstacle is closer
CRITICAL_DISTANCE = 10      # cm - immediate stop and reverse

# Object detection
DETECTION_INTERVAL = 2.0    # seconds between detection scans

# Audio settings
VOLUME_DB = 3
TTS_VOICE = 'echo'
VOICE_INSTRUCTIONS = "Speak with excitement when you see animals, be curious and friendly."
```

### Wake Words

| Action | Wake Words |
|--------|------------|
| Stop | "stop", "freeze", "halt" |
| Start | "start", "go", "move", "roam" |

### How It Works

1. **Obstacle Avoidance**: Uses ultrasonic sensor to detect obstacles
   - At critical distance (<10cm): Reverses and turns
   - At minimum distance (<25cm): Stops and looks left/right to find best path
   - Clear path: Proceeds forward with occasional random variations

2. **Object Detection**: Periodically captures camera frames and sends to GPT-4o Vision
   - Analyzes image for dogs, cats, people, and interesting objects
   - If something notable is detected, announces it via text-to-speech
   - LED lights up when animals are detected

3. **Voice/Keyboard Loop**: Continuously listens for commands
   - Wake words control roaming state
   - Other speech/text triggers conversation mode with context about recent detections

----------------------------------------------------------------

## Preset actions

### Preset actions

- `preset_actions.py` contains preset actions, such as `shake_head`, `nod`, `depressed`, `honking`, `start_engine`, etc. You can run this file to see the preset actions:</br>
  `python3 preset_actions.py`

