import cv2
from tracking.object_detector import ObjectDetector
from tracking.object_tracker import ObjectTracker
from utils.ball_speed_calculator import BallSpeedCalculator
from utils.game_stats import GameStats

# Load the pre-trained object detection and tracking models
detector_model = "models/detector_model.pth"
tracker_model = "models/tracker_model.pth"
object_detector = ObjectDetector(detector_model)
object_tracker = ObjectTracker(tracker_model)

# Create the ball speed calculator and game statistics tracker
ball_speed_calculator = BallSpeedCalculator()
game_stats = GameStats()

# Open the video file for analysis
video_file = "path/to/your/video/file.mp4"
cap = cv2.VideoCapture(video_file)

# Analyze each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and track the ball in the current frame
    ball_position = object_detector.detect(frame)
    ball_position = object_tracker.track(ball_position)

    # Calculate the speed of the ball and update the game statistics
    ball_speed = ball_speed_calculator.calculate_speed(ball_position)
    game_stats.update(ball_speed)

    # Draw the ball and its speed on the frame
    cv2.circle(frame, ball_position, 10, (0, 255, 0), -1)
    cv2.putText(frame, f"Ball speed: {ball_speed:.2f} m/s", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame and exit on key press
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture and destroy the window
cap.release()
cv2.destroyAllWindows()
