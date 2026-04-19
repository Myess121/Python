import cv2
import numpy as np
import torch
import time

from model_def import get_model,count_parameters
device=torch.device("cuda")
model_path='models_new/best_model.pth'
model=get_model(10).to(device)

checkpoint = torch.load(model_path,map_location=device)
if isinstance(checkpoint,dict):
    if 'model_state_dict' in checkpoint:
        state_dict=checkpoint['model_state_dict']
    else:
        state_dict=checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    total_params=count_parameters(model)
    test_input=torch.randn(1,1,28,28).to(device)
    with torch.no_grad():
        output=model(test_input)
        probability=torch.softmax(output,dim=1)
        confidence,prediction=torch.max(probability,1)
def preprocess_digit(image):
    if len(image.shape)==3:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray=image
    height,width=gray.shape
    center=(width//2,height//2)
    angle=3
    rotation_matrix=cv2.getRotationMatrix2D(center,angle,1.0)
    rotated=cv2.warpAffine(gray,rotation_matrix,(width,height),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
    blurred=cv2.GaussianBlur(rotated,(3,3),0)
    binary=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,5)
    kernel=np.ones((2,2),np.uint8)
    processed=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
    processed=cv2.morphologyEx(processed,cv2.MORPH_CLOSE,kernel)
    resized=cv2.resize(processed,(28,28),interpolation=cv2.INTER_LANCZOS4)
    coords=cv2.findNonZero(resized)
    if coords is not None:
        x,y,w,h=cv2.boundingRect(coords)
        offset_x=(28-w)//2-x
        offset_y=(28-h)//2-y
        m=np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        resized=cv2.warpAffine(resized, m, (28, 28), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        normalized=resized.astype('float32') / 255.0
        mean=0.1307
        std=0.3081
        normalized=(normalized-mean)/std
        tensor_image=torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
        return tensor_image, resized
def find_digits(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(5,5),0)
    binary1=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    _,binary2=cv2.threshold(blurred,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    kernel=np.ones((3,3),np.uint8)
    combined=cv2.bitwise_or(binary1,binary2)
    cleaned=cv2.morphologyEx(combined,cv2.MORPH_CLOSE,kernel)
    cleaned=cv2.morphologyEx(cleaned,cv2.MORPH_OPEN,kernel)
    contours,_=cv2.findContours(cleaned,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digit_contours=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        x,y,w,h=cv2.boundingRect(contour)
        if area > 100 and 10 < w < 200 and 10 < h < 200 and 0.2 < w / h < 4.0:
            digit_contours.append(contour)
    return digit_contours, cleaned

def predict_digit(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = torch.softmax(output,dim=1)
        confidence,prediction = torch.max(probabilities,1)
        return prediction.item(),confidence.item()
cap=cv2.VideoCapture(0)
confidence_threshold=0.8
frame_count=0
start_time=time.time()
last_detections=[]
while True:
    ret,frame=cap.read()
    frame_count+=1
    height,width=frame.shape[:2]
    roi_size=400
    x1=(width-roi_size)//2
    y1=(height-roi_size)//2
    x2=x1+roi_size
    y2=y1+roi_size
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(frame,"",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    roi=frame[y1:y2,x1:x2]
    current_detections=[]
    if frame_count%5==0:
        contours,thresh=find_digits(roi)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            padding=20
            x=max(0,x-padding)
            y=max(0,y-padding)
            w=min(roi_size-x,w+2*padding)
            h=min(roi_size-y,h+2*padding)
            digit_roi=roi[y:y+h,x:x+w]
            if digit_roi.size>0 and w>20 and h>20:
                try:
                    processed_tensor,processed_img=preprocess_digit(digit_roi)
                    digit, confidence=predict_digit(processed_tensor)

                    if confidence>confidence_threshold:
                        current_detections.append((x,y,w,h,digit,confidence))
                except Exception as e:
                    pass
        last_detections=current_detections

    for (x,y,w,h,digit,confidence) in last_detections:
        if confidence>0.9:
            color=(0,255,0)
        elif confidence>0.7:
            color=(0,165,255)
        else:
            color=(0,0,255)
        cv2.rectangle(roi,(x,y),(x+w,y+h),color,2)
        label=f"{digit} ({confidence:.2f})"
        cv2.putText(roi,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if len(last_detections) > 0:
        try:
            _, thresh_display = find_digits(roi)
            thresh_display = cv2.cvtColor(thresh_display, cv2.COLOR_GRAY2BGR)
            thresh_display = cv2.resize(thresh_display, (200, 200))
            frame[10:210, 10:210] = thresh_display
            cv2.putText(frame, "Processed View", (10, 230),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except:
            pass
    cv2.imshow('Digit Recognition', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = int(time.time())
        filename = f'digit_capture_{timestamp}.jpg'
        cv2.imwrite(filename, frame)
    elif key == ord('+'):
        confidence_threshold = min(0.95, confidence_threshold + 0.05)
    elif key == ord('-'):
        confidence_threshold = max(0.1, confidence_threshold - 0.05)
    elif key == ord('c'):
        last_detections = []

cap.release()
cv2.destroyAllWindows()