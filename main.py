import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapipe


 

class Pose_Aware:
    # print(df)
    def __init__(self) -> None:
       
        self.numbers = [234, 50, 36, 49, 45, 4, 275, 279, 266, 280, 454,323,361,435,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234]
        self.pairs = [[self.numbers[i], self.numbers[i + 1]] for i in range(0, len(self.numbers) - 1)]
        self.mp_face_mesh = mediapipe.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True)
        face_oval = self.mp_face_mesh.FACEMESH_FACE_OVAL

    def generate_mask(self,img,face_crop=False,image_size=(256,256),crop_margin=0.1):
        routes = []
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        landmarks = results.multi_face_landmarks[0]
        #crop the face from the image
        if face_crop:
            x_min = int(min([landmark.x for landmark in landmarks.landmark]) * img.shape[1])
            x_max = int(max([landmark.x for landmark in landmarks.landmark]) * img.shape[1])
            y_min = int(min([landmark.y for landmark in landmarks.landmark]) * img.shape[0])
            y_max = int(max([landmark.y for landmark in landmarks.landmark]) * img.shape[0])
            x_max = min(x_max + int(crop_margin * (x_max - x_min)), img.shape[1])
            x_min = max(x_min - int(crop_margin * (x_max - x_min)), 0)
            y_max = min(y_max + int(crop_margin * (y_max - y_min)), img.shape[0])
            y_min = max(y_min - int(crop_margin * (y_max - y_min)), 0)

            # Calculate width and height maintaining aspect ratio
            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = float(image_size[0]) / float(image_size[1]) 
            
            # Adjust the bounding box to maintain the specified aspect ratio
            adjusted_width = x_max - x_min
            adjusted_height = int(adjusted_width / aspect_ratio)
            y_max = min(y_max, y_min + adjusted_height)
            img = img[y_min:y_max, x_min:x_max]
            results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            landmarks = results.multi_face_landmarks[0]

        for source_idx, target_idx in self.pairs:
            # print(f"Draw a line between {source_idx}th landmark point to {target_idx}th landmark point")
            source = landmarks.landmark[source_idx]
            target = landmarks.landmark[target_idx]
            relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
            relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))
            routes.append(relative_source)
            routes.append(relative_target)
        mask = np.zeros_like(img)
        polygon_points = np.array(routes, dtype=np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
        #resize the image
        img = cv2.resize(img, image_size)
        mask = cv2.resize(mask, image_size)

        return img, mask


make_mask = Pose_Aware()
video = cv2.VideoCapture("./3.mp4")
success, img = video.read()
idx=1
while success:
    success, img = video.read()
    if not success:
        break
    image_size = (256,256)
    crop_margin = 0.2
    img,mask = make_mask.generate_mask(img,face_crop=True,image_size=image_size,crop_margin=crop_margin)
    mask_name =f"{idx:05d}_front_mask.jpg"
    frame_name =f"{idx:05d}.jpg"
    cv2.imwrite(f"/Users/ameerazam/Test-Gan/Post-aware/mask/{mask_name}",mask)
    cv2.imwrite(f"/Users/ameerazam/Test-Gan/Post-aware/frame/{frame_name}",img)
    idx+=1
    
