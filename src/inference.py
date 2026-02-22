import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Load the best model
latest_train = max(glob.glob('runs/detect/train*'), key=os.path.getmtime)
custom_model = YOLO(os.path.join(latest_train, 'weights', 'best.pt'))

# Generate a new ball image to test
test_img = np.zeros((640,640,3), dtype=np.uint8)
cv2.circle(test_img, (300,300), 50, (0,0,255), -1)
cv2.imwrite('test_ball.jpg', test_img)

# Predict
results = custom_model.predict('test_ball.jpg', save=True, conf=0.2)

# Show predictions
latest_predict = max(glob.glob('runs/detect/predict*'), key=os.path.getmtime)
result_path = os.path.join(latest_predict, 'test_ball.jpg')

img = mpimg.imread(result_path)
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis('off')
plt.title("Did it find the ball?")
plt.show()