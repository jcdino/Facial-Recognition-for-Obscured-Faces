from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import cv2 as cv
import numpy as np
import os
import time
import dlib
import math
from imutils import paths
import imutils
import random

class dectecting_functions :
	def detect_and_predict_accesory(frame, faceNet, accesoryNet):
		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
			(104.0, 177.0, 123.0))

		faceNet.setInput(blob)
		detections = faceNet.forward()

		faces = []
		preds = []

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				face = frame[startY:endY, startX:endX]

				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				face = np.expand_dims(face, axis=0)

				faces.append(face)

		if len(faces) > 0:
			preds = accesoryNet.predict(faces)
		return preds


class recognizing_functions :

	def face_recognition(frame, faceNet, embedder, recognizer, le, accesoryList, recorded_time, acc):
		(h, w) = frame.shape[:2]
		imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
										  (104.0, 177.0, 123.0), swapRB=False, crop=False)

		faceNet.setInput(imageBlob)
		detections = faceNet.forward()

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
				if fW < 20 or fH < 20:
					continue
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0),
												 swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				saved_time = time.time()
				text = "{}: {:.2f}%".format(name, proba * 100)
				#text = "status : {}".format(acc)

				if accesoryList != "fullface":
					if name[0:7] != "unknown":
						if name in recorded_time["name"]:
							if saved_time - recorded_time["time"][recorded_time["name"].index(name)] > 2:
								if len(os.walk("saved_samples"+"/"+name+"/").__next__()[2]) < 10:
									cv2.imwrite("saved_samples"+"/"+name
												+ "/" + str(len(os.walk("saved_samples"+"/"+name +"/").__next__()[2]))
												+ ".jpg", face)
									recorded_time["time"][recorded_time["name"].index(name)] = saved_time
						elif name not in recorded_time["name"]:
							if os.path.exists("saved_samples" + "/" + name + "/") == False :
								os.mkdir("saved_samples" + "/" + name + "/")
							if len(os.walk("saved_samples" + "/" + name + "/").__next__()[2]) < 10:
								cv2.imwrite("saved_samples" + "/" + name + "/" + "0" + ".jpg", face)
								recorded_time["name"].append(name)
								recorded_time["time"].append(saved_time)

				y = startY - 10 if startY - 10 > 10 else startY + 10
				accesoryList["count"][accesoryList["name"].index(name)] += 1
				if name[-4:] == "mask":
					for i in range(0, len(le.classes_)):
						accesoryList["avg"][accesoryList["name"].index(le.classes_[i])] += preds[i]
				else:
					accesoryList["avg"][accesoryList["name"].index(name)] += proba

				if accesoryList["best"][accesoryList["name"].index(name)] < proba:
					accesoryList["best"][accesoryList["name"].index(name)] = proba
				#cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
				if name == "unknown_mask":
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
				else:
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)


	def second_recognition(faceNet, embedder, recognizer, le, score_list):
		path = "fused_pic/"
		imagePaths = list(paths.list_images(path))
		for (n, imagePath) in enumerate(imagePaths):
			img = cv2.imread(imagePath)
			frame = imutils.resize(img, width=600)
			(h, w) = frame.shape[:2]
			imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
											  (104.0, 177.0, 123.0), swapRB=False, crop=False)
			faceNet.setInput(imageBlob)
			detections = faceNet.forward()
			for i in range(0, detections.shape[2]):
				if detections[0, 0, i, 2] > 0.5:
					j = i
			box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(96, 96), (0, 0, 0),
											 swapRB=True, crop=False)

			embedder.setInput(faceBlob)
			vec = embedder.forward()

			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			detected_name = le.classes_[j]
			"""
						if detected_name == imagePath.split("/")[1].split("_")[0]:
				print("{} == {}({})".format(imagePath.split("/")[1].split("_")[0],detected_name,proba))
			else:
				print("{} != {}".format(imagePath.split("/")[1].split("_")[0], detected_name))
			"""

			if detected_name in score_list["name"]:
				if detected_name == imagePath.split("/")[1].split("_")[0]:
					score_list["count"][score_list["name"].index(detected_name)] += 1
					score_list["avg"][score_list["name"].index(detected_name)] += proba
					if score_list["best"][score_list["name"].index(detected_name)] < proba:
						score_list["best"][score_list["name"].index(detected_name)] = proba
				else:
					score_list["wrong"][score_list["name"].index(
						imagePath.split("/")[1].split("_")[0])] += 1


class fusion_functions:

	def mask_fuse(second_list,name,cnt,N):
		img_names = []
		path = "userface/"
		imagePaths = list(paths.list_images(path))
		for pic_names in enumerate(imagePaths):
			if name == str(pic_names).split("_")[0].split("/")[1]:
				img_names.append(pic_names[1])

		ran = random.choice(cnt)
		cnt.remove(ran)
		for back_pic in img_names:
			if back_pic.split("_")[1].split(".")[0] == str(ran):
				img = cv.imread(back_pic)
				break

		EYE = list(range(36, 48))

		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		lab_planes = cv2.split(lab)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		lab_planes[0] = clahe.apply(lab_planes[0])
		lab = cv2.merge(lab_planes)
		img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

		path = "saved_samples/" + name + "_mask"
		imagePaths = list(paths.list_images(path))
		for (n, imagePath) in enumerate(imagePaths):
			print("[INFO] for {} processing mask image {}/{}".format(name, n, len(imagePaths)-1))
			img2 = cv.imread(imagePath)
			lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
			lab_planes2 = cv2.split(lab2)
			clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			lab_planes2[0] = clahe2.apply(lab_planes2[0])
			lab2 = cv2.merge(lab_planes2)
			img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
			img2_s = img2

			img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			dets = detector(img_gray, 1)

			for face in dets:
				shape = predictor(img, face)
				list_points = []
				for p in shape.parts():
					list_points.append([p.x, p.y])
				list_points = np.array(list_points)

			for i, pt in enumerate(list_points[EYE]):
				pt_pos = (pt[0], pt[1])

				if i == 0:
					eye_left = pt_pos
				elif i == 9:
					eye_right = pt_pos

			w1 = abs(eye_left[0] - eye_right[0])

			if eye_right[1] > eye_left[1]:
				h1 = abs(eye_left[1] - eye_right[1])
				higher_eye = "left_eye"
			else:
				h1 = abs(eye_right[1] - eye_left[1])
				higher_eye = "right_eye"

			b = (w1 ** 2 + h1 ** 2) ** 0.5
			A = math.degrees(math.acos(w1 / b))
			h2, w2 = img.shape[:2]
			if higher_eye == "left_eye":
				M1 = cv.getRotationMatrix2D((w2 / 2, h2 / 2), A, 1)
				img_angle = cv.warpAffine(img, M1, (w2, h2))
			else:
				M1 = cv.getRotationMatrix2D((w2 / 2, h2 / 2), -A, 1)
				img_angle = cv.warpAffine(img, M1, (w2, h2))
			img3 = img_angle



			img_angle_gray = cv.cvtColor(img_angle, cv.COLOR_BGR2GRAY)
			dets2 = detector(img_angle_gray, 1)

			for face in dets2:
				shape1 = predictor(img, face)
				list_points2 = []

			for p in shape1.parts():
				list_points2.append([p.x, p.y])

			list_points2 = np.array(list_points2)

			faces = detector(img3)
			for face in faces:
				y_top=face.top()
				landmarks = predictor(img3, face)
				x1 = []
				y1 = []
				for n in range(0, 68):
					x1.append(landmarks.part(n).x)
					y1.append(landmarks.part(n).y)

				wc = abs(x1[17] - x1[26])
				h_f = abs(y_top - y1[28])
				y_top1 = y_top - int(h_f / 2)
				hc = abs(y_top1 - y1[28])

				crop1 = img_angle[y_top1:y_top1 + hc, x1[17]:x1[17] + wc]

				x_left = list_points2[1][0]
				y_top = list_points2[29][1]
				x_right = list_points2[15][0]
				y_bottom = list_points2[8][1]
				w = abs(x_left - x_right)
				h = abs(y_top - y_bottom)
				path_img = img3
				crop_img = path_img[y_top:y_top + h, x_left:x_left + w]

				img2_gray = cv2.cvtColor(img2_s, cv2.COLOR_BGR2GRAY)
				[h2, w2] = img2_gray.shape
				irr = 0
				a= 0
				half_h = int(h2 / 2)
				err = 'Yes'
				while err=="Yes":
					img5 = img2_s
					h_crop = half_h + a
					specs = cv.resize(crop_img, (w2, int(h_crop)))
					specs_gray = cv.cvtColor(specs, cv.COLOR_BGR2GRAY)
					h3, w3 = specs_gray.shape

					for i in range(0, h3):
						for j in range(0, w3):
							if specs[i, j][2] != 0:
								img5[i - h3, j - w3] = specs[i, j]
					fake_out = cv.copyMakeBorder(img5, 20, 20, 20, 20,cv.BORDER_CONSTANT, value=[0, 0, 0])
					faces1 = detector(fake_out)
					irr += 1
					if not faces1:
						print("recalculating coordinate",irr)
						if irr == 6:
							a = 0
						if irr < 6:
							a += half_h/40
						elif irr >= 6:
							a -= half_h/40
						if irr > 9:
							break
					else:
						err='No'
			if err == "No":
				for face in faces1:
					y1_1 = face.top()
					landmarks = predictor(fake_out, face)
					x = []; y = []
					for n in range(0, 68):
						x.append(landmarks.part(n).x); y.append(landmarks.part(n).y)

					h_f2 = abs(y1_1 - y[28]); y_top2 = y1_1 - int(h_f2 / 2)
					hc2 = abs(y_top2 - y[28]); wc2 = abs(x[17] - x[26])

					crop2 = fake_out[y_top2:y_top2 + hc2, x[17]:x[17] + wc2]
				if 0 not in crop2.shape:
					h_c1, w_c1, ch1 = crop1.shape
					crop2 = cv2.resize(crop2, (w_c1, h_c1))

					for i in range(0, h_c1):
						for j in range(0, w_c1):
							if crop2[i, j][2] != 0:
								img_angle[y_top1 + i, x1[17] + j] = crop2[i, j]

					img4 = cv.copyMakeBorder(img_angle, 100, 100, 100, 100, cv.BORDER_CONSTANT, value=[0, 0, 0])
					cv2.imwrite("fused_pic/"+name+"_"+str(N)+".jpg", img4)
					second_list["total"][second_list["name"].index(name)] += 1
					N += 1
				else:
					print("<<<<ERROR>>>> crop2 has size error")
			else:
				print("<<<<ERROR>>>>",imagePath.split("/")[1],"has error")
			if N > 19:
				break
		return N, cnt