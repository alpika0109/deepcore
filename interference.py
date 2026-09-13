import cv2

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break

    # Get the prediction from the dual view model
    pred_class = preprocess_and_predict(frame1, frame2)
    
    # (Optionally) Overlay the prediction on the combined frame
    combined_frame = cv2.hconcat([frame1, frame2])
    cv2.putText(combined_frame, f"Prediction: {pred_class}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    cv2.imshow("Dual Camera View", combined_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
