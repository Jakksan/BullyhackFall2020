import cv2

original = cv2.imread("/Users/jacksondegruiter/Documents/Bullyhack/python/dataset/images/User.1.1.jpg")
resized = cv2.resize(original, dsize=(100,100))
count=0
while(count < 10000):
    cv2.imshow("Resize Test", resized)
    count+=1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if count>10000:
        break
cv2.destroyAllWindows()
