import cv2
import os

DATA_DIR = './images'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
key=113
number_of_sign =10
data_size = 50
label={"0":"A",
       "1":"B",
       "2":"C",
       "3":"D",
       "4":"E",
       "5":"F",
       "6":"O","7":"1","8":"2","9":"3"}

cap = cv2.VideoCapture(0)
for j in range(number_of_sign):
    if not os.path.exists(os.path.join(DATA_DIR,str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))
    print('Collecting data for class {}'.format(label[str(j)]))

    while True:
         ret, frame = cap.read()
         cv2.rectangle(frame, (20, 20), (700, 70), (0,0,0), -1)
         cv2.putText(frame, 'press q to start collection:{}'.format(label[str(j)]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
         cv2.imshow("Data Collection", frame)
         key = cv2.waitKey(10)
         if key == ord('q') or key == ord('s'):
             break

    if key == ord('q'):
              counter = 0

              n = 50
              while counter < data_size:
                  ret, frame = cap.read()
                  cv2.rectangle(frame, (20, 20), (600, 80), (0, 0, 0), -1)
                  cv2.putText(frame, 'Collecting.... {}:)'.format(n), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 2, cv2.LINE_AA)
                  counter += 1
                  cv2.imwrite(os.path.join(DATA_DIR,str(j), '{}.jpg'.format(counter)), frame)
                  cv2.imshow("Data Collection", frame)
                  n=n-1
                  cv2.waitKey(100)
    if key == ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            break

cap.release()
cv2.destroyAllWindows()
